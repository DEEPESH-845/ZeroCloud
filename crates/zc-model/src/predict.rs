//! Turning measured hardware into predicted performance.

use crate::fit::{Confidence, Fit};
use crate::kv::KvPrecision;
use crate::spec::{ModelSpec, Quant};

/// Which compute backend the runtime will actually use.
///
/// This changes which bandwidth figure is correct, which is easy to get wrong:
/// on unified-memory Apple Silicon the GPU saturates the memory controller
/// substantially better than the CPU cluster can, so predicting Metal
/// inference from a CPU-core bandwidth measurement understates it badly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// CPU only. Use the performance-core bandwidth figure.
    Cpu,
    /// Apple Metal on unified memory. Use the all-core peak as a proxy for
    /// GPU-achievable bandwidth — both land near 85% of theoretical.
    Metal,
    /// Discrete GPU. Weights in VRAM run at VRAM bandwidth; the rest spills
    /// to host RAM, where the CPU runs those layers at host bandwidth — which
    /// is what every runtime does on a partial offload.
    Discrete,
}

/// Measured machine characteristics. Every field comes from `zc-bench`;
/// nothing here is looked up from a spec sheet.
#[derive(Debug, Clone)]
pub struct Hardware {
    pub backend: Backend,
    /// Read bandwidth achievable by the cores inference will actually use.
    ///
    /// Deliberately the performance-core figure, not the all-core peak.
    /// Efficiency cores raise raw bandwidth (more loads in flight) but lose it
    /// again in real inference, where dequantisation is real work and slow
    /// cores straggle at the join barrier.
    pub ram_bw_gbs: f64,
    /// All-core peak, for reporting only.
    pub ram_bw_peak_gbs: f64,
    /// Random 128 KiB read at queue depth, uncached. The relevant figure for
    /// streaming weights, which is neither purely sequential nor QD1.
    pub disk_gbs: f64,
    /// Peak FMA throughput, all inference threads.
    pub gflops: f64,
    /// Bytes we may actually allocate.
    pub usable_bytes: u64,
    pub threads: u32,
    /// Dedicated VRAM of the single card we would run on, in bytes. 0 when
    /// there is no discrete GPU.
    ///
    /// One card, never the sum: splitting a model across cards is a different
    /// execution model with its own interconnect cost, and pretending two 8 GB
    /// cards are a 16 GB card predicts "fits" for something that does not.
    pub vram_bytes: u64,
    /// VRAM read bandwidth, GB/s.
    pub vram_bw_gbs: f64,
    /// Whether `vram_bw_gbs` was measured on this machine or looked up from a
    /// table of card families.
    ///
    /// Load-bearing: everything else in `Hardware` is measured, and a looked-up
    /// number must not be able to earn the same confidence as a measured one.
    /// `predict_with` caps confidence when this is false.
    pub vram_bw_measured: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Comfortable for interactive chat.
    Good,
    /// Works, but you will notice the wait.
    Usable,
    /// Batch work only — queue it and come back.
    Slow,
    /// Cannot run: no context left after the weights.
    WontFit,
}

#[derive(Debug, Clone)]
pub struct Prediction {
    /// Fraction of active weights that fit in RAM. 1.0 means no streaming.
    pub resident_fraction: f64,
    /// Predicted decode speed as a (low, high) range. We never report a point
    /// estimate: a single number reads as a promise, and a missed promise is
    /// what destroys trust in a prediction tool.
    pub decode_tok_s: (f64, f64),
    /// Prompt processing speed, tokens/sec.
    ///
    /// `None` when no run has been measured on this backend. There is no
    /// prior: our compute benchmark cannot see the GPU and cannot reach the
    /// int8 kernels real prefill uses, so any derived figure would be wrong by
    /// an unknown factor. Reporting unknown beats reporting confidently wrong.
    pub prefill_tok_s: Option<f64>,
    /// Time to first token for a prompt of this length, seconds. `None` for
    /// the same reason as `prefill_tok_s`.
    pub ttft_s: Option<f64>,
    /// Evidence behind the prefill figure.
    pub prefill_confidence: Confidence,
    pub max_context: u32,
    pub kv_bytes_per_token: u64,
    pub verdict: Verdict,
    /// Seconds per token if the machine achieved 100% of measured bandwidth.
    ///
    /// Exposed so `zc verify` can invert the model: given a real measurement,
    /// `implied_eta = actual_tok_s * raw_seconds_per_token`. That single number
    /// is what the whole calibration dataset accumulates.
    pub raw_seconds_per_token: f64,
    /// The efficiency we assumed, so verify can report assumed vs implied.
    pub assumed_eta: f64,
    /// How much evidence stands behind `assumed_eta`. Shown to the user, so a
    /// number backed by 40 real runs is visibly different from a shipped guess.
    pub confidence: Confidence,
}

/// Memory available to us *right now*, given whatever else is running.
///
/// Starts from *available* (not total, not free) and holds back a reserve so
/// the machine stays usable — a user who cannot open a browser while the model
/// runs will uninstall regardless of how fast it was.
pub fn current_budget(total: u64, available: u64, ceiling: Option<u64>, reserved: u64) -> u64 {
    let base = ceiling.map_or(available, |c| available.min(c));
    let reserve = (total / 10).max(1 << 30); // 10% of RAM, at least 1 GiB
    base.saturating_sub(reserve).saturating_sub(reserved)
}

/// Memory available on a machine that is otherwise idle.
///
/// This, not `current_budget`, is the right basis for "what can this machine
/// run?" — the user asking is deciding what to download, and by the time they
/// run it they will have closed the browser. Predicting against instantaneous
/// free memory answers a question nobody asked and reports failure on machines
/// that are perfectly capable.
///
/// The floor covers the OS plus a shell and a terminal. It is deliberately
/// generous: predicting a model fits and having it OOM is far worse than
/// predicting it does not and being pleasantly surprised.
pub fn potential_budget(total: u64, ceiling: Option<u64>, reserved: u64) -> u64 {
    let base = ceiling.map_or(total, |c| total.min(c));
    let os_floor = if cfg!(target_os = "macos") {
        // macOS wires down noticeably more, and compression makes reclaim
        // less predictable.
        (base / 5).max(3 << 30)
    } else if cfg!(target_os = "windows") {
        (base / 5).max(5 * (1 << 29)) // 2.5 GiB
    } else {
        (base / 6).max(3 * (1 << 29)) // 1.5 GiB
    };
    base.saturating_sub(os_floor).saturating_sub(reserved)
}

/// Fraction of peak read bandwidth a real inference loop achieves.
///
/// Below 1.0 because decode does more than stream: dequantisation, attention,
/// sampling, and thread synchronisation all consume time our pure-read
/// benchmark does not. A primary calibration target.
const DECODE_EFFICIENCY: f64 = 0.70;

/// Predict using shipped priors only.
pub fn predict(
    spec: &ModelSpec,
    quant: &Quant,
    hw: &Hardware,
    kv_prec: KvPrecision,
    prompt_tokens: u32,
    ubatch: u32,
) -> Prediction {
    predict_with(spec, quant, hw, kv_prec, prompt_tokens, ubatch, &Fit::default())
}

/// Predict using whatever calibration evidence exists.
///
/// The fit supplies both the coefficient and the range width. A bucket backed
/// by many consistent runs narrows the range; a bucket with none falls back to
/// the prior and stays wide. The range is the promise we make to the user, so
/// it must reflect what we actually know.
#[allow(clippy::too_many_arguments)]
pub fn predict_with(
    spec: &ModelSpec,
    quant: &Quant,
    hw: &Hardware,
    kv_prec: KvPrecision,
    prompt_tokens: u32,
    ubatch: u32,
    fit: &Fit,
) -> Prediction {
    // Budget policy. KV and compute buffers are *mandatory* resident: they are
    // written every token and cannot be streamed. Weights are read-only and can
    // be paged from disk, so they must never block context outright — they
    // trade against it.
    //
    // We therefore reserve room for a minimum usable context first, then let
    // the weights cache take whatever remains. A model you cannot hold a
    // conversation with is useless however fast it decodes.
    const MIN_USABLE_CTX: u32 = 2048;
    let compute_buf = spec.compute_buffer_bytes(ubatch);
    let kv_floor = spec.kv_bytes(MIN_USABLE_CTX, kv_prec);

    // VRAM is capacity in exactly the same sense host RAM is: it holds weights
    // and KV that would otherwise stream from disk. So it joins the pool the
    // budget is drawn from, and only the *speed* of what lands there differs.
    let pool = hw.usable_bytes.saturating_add(hw.vram_bytes);
    let after_bufs = pool.saturating_sub(compute_buf);
    let weights_cached = quant.bytes.min(after_bufs.saturating_sub(kv_floor));
    let for_kv = after_bufs.saturating_sub(weights_cached);
    // Memory is not the only ceiling. A sliding-window model stops accumulating
    // KV once the window fills, so the budget alone would report a context the
    // model was never trained to handle.
    let max_context = spec
        .max_context_in(for_kv, kv_prec)
        .min(spec.n_ctx_train.unwrap_or(u32::MAX));

    // Operating point for the speed estimate: a realistic conversation length,
    // not the theoretical ceiling.
    let reserve_ctx = max_context.clamp(512, 4096);

    // Residency, MoE-aware and tiered.
    //
    // The hot core (embeddings, attention, routers, shared experts) is read on
    // every token, so it is pinned in the fastest tier first and is the most
    // valuable thing in memory. Whatever remains caches routed experts.
    // Computing one blended fraction over *total* bytes and applying it to
    // *active* bytes understates MoE badly, because the hot core is a small
    // slice of the total but a large slice of the active set.
    //
    // ponytail: assumes uniform expert access. Real routing is heavily skewed,
    // which makes a cache of a given size hit more often than this predicts —
    // so the estimate is conservative. Replace with a measured hit-rate curve
    // once calibration records carry one.
    let total_bytes = quant.bytes as f64;
    let hot = spec.hot_core_bytes(quant) as f64;
    let cold_total = (total_bytes - hot).max(0.0);
    let active = spec.active_bytes(quant) as f64;
    let active_cold = (active - hot).max(0.0);

    // KV and the compute buffers are written every token, so on a discrete GPU
    // they occupy VRAM before any weight does. Charged at the operating
    // context, which is what a real session allocates — charging the *maximum*
    // context would starve the card of weights on a machine with plenty of RAM.
    let vram_overhead = (compute_buf + spec.kv_bytes(reserve_ctx, kv_prec)).min(hw.vram_bytes);
    let vram_for_weights = (hw.vram_bytes - vram_overhead).min(weights_cached);

    // Fastest tier first, which is what every runtime does. With no GPU the
    // first tier is empty and this reduces exactly to the host-RAM-then-disk
    // model it replaces.
    let tiers = [
        (vram_for_weights as f64, hw.vram_bw_gbs),
        ((weights_cached - vram_for_weights) as f64, hw.ram_bw_gbs),
    ];
    let mut hot_left = hot;
    let mut cold_left = cold_total;
    let mut seconds_per_token = 0.0;
    let mut from_fast = 0.0;
    for (capacity, gbs) in tiers {
        if capacity <= 0.0 || gbs <= 0.0 {
            continue;
        }
        let pinned = capacity.min(hot_left);
        hot_left -= pinned;
        let cached = (capacity - pinned).min(cold_left);
        cold_left -= cached;
        // Active bytes this tier serves: all of the hot core it holds, plus the
        // share of the cold set it caches.
        let served = pinned
            + if cold_total > 0.0 {
                active_cold * (cached / cold_total)
            } else {
                0.0
            };
        seconds_per_token += served / (gbs * 1e9);
        from_fast += served;
    }
    let from_disk = (active - from_fast).max(0.0);
    seconds_per_token += from_disk / (hw.disk_gbs.max(0.01) * 1e9);

    // Reported as a fraction of *active* bytes — what is actually read per
    // token — rather than of the whole file.
    let resident_fraction = if active > 0.0 { from_fast / active } else { 1.0 };

    // Shipped prior, used only when no calibration record matches.
    let prior = DECODE_EFFICIENCY * quant.family.efficiency();
    let mut coef = fit.lookup(&format!("{:?}", hw.backend), quant.family, prior);

    // A prediction is only as trustworthy as its slowest-moving input. When
    // the weights sit in VRAM whose bandwidth we looked up in a table instead
    // of measuring, the number underneath the range is not a measurement, and
    // it must not be able to earn `high` however many runs back the
    // coefficient — those runs came from other machines with other cards.
    // Erring wide is the house rule: a confident wrong number is what kills a
    // prediction tool.
    if vram_for_weights > 0 && !hw.vram_bw_measured {
        coef.confidence = match coef.confidence {
            Confidence::High | Confidence::Medium => Confidence::Low,
            other => other,
        };
        coef.spread = coef.spread.max(coef.confidence.min_spread());
    }
    let eta = coef.eta;
    let decode = if seconds_per_token > 0.0 {
        eta / seconds_per_token
    } else {
        0.0
    };

    // Prefill processes the whole prompt in one weight pass, so it is bound by
    // arithmetic, not bandwidth: 2 flops per parameter per token.
    //
    // The scale factor is measured, never assumed. It absorbs three things at
    // once — GEMM versus peak FMA, int8 versus f32, and GPU versus CPU — whose
    // product spans more than an order of magnitude. That is far too much to
    // guess, so with no measurement we say so.
    let flops_per_token = 2.0 * spec.active_params() as f64;
    let prefill_coef = fit.prefill_scale(&format!("{:?}", hw.backend));
    let prefill = prefill_coef
        .as_ref()
        .map(|c| hw.gflops * 1e9 * c.eta / flops_per_token);

    // WontFit means genuinely cannot run: not even a minimal context fits
    // alongside the compute buffers. A model larger than RAM is not "won't
    // fit" — it streams, and the honest answer is a slow speed, not a refusal.
    let verdict = if max_context < 512 {
        Verdict::WontFit
    } else if decode >= 10.0 {
        Verdict::Good
    } else if decode >= 3.0 {
        Verdict::Usable
    } else {
        Verdict::Slow
    };

    Prediction {
        resident_fraction,
        decode_tok_s: (
            // A spread wider than 100% is possible and legitimate -- it is what
            // a bucket holding two machines that disagree by 8x should report.
            // What it must never do is print a negative rate: nothing generates
            // fewer than zero tokens per second, and `-0.2-0.7` does not even
            // parse as a range. Zero is the honest floor, and it still reads as
            // "we cannot promise this runs usefully at all".
            (decode * (1.0 - coef.spread)).max(0.0),
            decode * (1.0 + coef.spread),
        ),
        prefill_tok_s: prefill,
        ttft_s: prefill.map(|p| prompt_tokens as f64 / p.max(1e-6)),
        prefill_confidence: prefill_coef
            .map_or(Confidence::Prior, |c| c.confidence),
        max_context,
        kv_bytes_per_token: spec.kv_bytes_per_token(reserve_ctx, kv_prec),
        verdict,
        raw_seconds_per_token: seconds_per_token,
        assumed_eta: eta,
        confidence: coef.confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{Attention, Moe};

    fn llama3_8b() -> ModelSpec {
        ModelSpec {
            id: "llama-3.1-8b".into(),
            n_layers: 32,
            n_embd: 4096,
            n_vocab: 128256,
            params: 8_030_000_000,
            attention: Attention::Gqa { n_kv_heads: 8, head_dim: 128 },
            moe: None,
            n_ctx_train: None,
            quants: vec![],
        }
    }

    fn llama3_70b() -> ModelSpec {
        ModelSpec {
            id: "llama-3.1-70b".into(),
            n_layers: 80,
            n_embd: 8192,
            n_vocab: 128256,
            params: 70_600_000_000,
            attention: Attention::Gqa { n_kv_heads: 8, head_dim: 128 },
            moe: None,
            n_ctx_train: None,
            quants: vec![],
        }
    }

    /// Hand-computed: 2 (K and V) * 32 layers * 8 kv heads * 128 dim * 2 bytes
    /// = 131072 bytes = exactly 128 KiB per token.
    #[test]
    fn llama3_8b_kv_is_128kib_per_token() {
        let m = llama3_8b();
        assert_eq!(m.kv_bytes(1, KvPrecision::F16), 128 * 1024);
        // And therefore exactly 1 GiB at 8K context.
        assert_eq!(m.kv_bytes(8192, KvPrecision::F16), 1 << 30);
    }

    /// The headline figure from the spec review: 2 * 80 * 8 * 128 * 2 = 320 KiB
    /// per token, so 32K context needs exactly 10 GiB of KV alone — more than
    /// the entire machine we target.
    #[test]
    fn llama3_70b_kv_at_32k_is_10gib() {
        let m = llama3_70b();
        assert_eq!(m.kv_bytes(1, KvPrecision::F16), 320 * 1024);
        assert_eq!(m.kv_bytes(32768, KvPrecision::F16), 10 << 30);
    }

    /// Q8 should be ~2x smaller and Q4 ~4x (slightly worse, due to scales).
    #[test]
    fn kv_quantisation_scales_as_expected() {
        let m = llama3_70b();
        let f16 = m.kv_bytes(4096, KvPrecision::F16) as f64;
        let q8 = m.kv_bytes(4096, KvPrecision::Q8) as f64;
        let q4 = m.kv_bytes(4096, KvPrecision::Q4) as f64;
        assert!((f16 / q8 - 1.882).abs() < 0.01, "q8 ratio {}", f16 / q8);
        assert!((f16 / q4 - 3.556).abs() < 0.01, "q4 ratio {}", f16 / q4);
    }

    /// MLA stores one compressed latent per layer instead of full K and V.
    /// DeepSeek-V3: 61 layers, kv_lora_rank 512, rope dim 64
    ///   -> 61 * 576 * 2 = 70272 bytes/token for a 671B model,
    /// versus 320 KiB/token for a 70B GQA model. An order of magnitude less
    /// KV for ten times the parameters.
    #[test]
    fn mla_kv_is_far_smaller_than_gqa() {
        let dsv3 = ModelSpec {
            id: "deepseek-v3".into(),
            n_layers: 61,
            n_embd: 7168,
            n_vocab: 129280,
            params: 671_000_000_000,
            attention: Attention::Mla { kv_lora_rank: 512, rope_head_dim: 64 },
            moe: Some(Moe {
                n_expert: 256,
                n_active: 8,
                expert_params: 44_000_000,
                shared_params: 17_000_000_000,
            }),
            n_ctx_train: None,
            quants: vec![],
        };
        assert_eq!(dsv3.kv_bytes(1, KvPrecision::F16), 70_272);
        assert!(dsv3.kv_bytes(1, KvPrecision::F16) < llama3_70b().kv_bytes(1, KvPrecision::F16));
    }

    /// Sliding-window layers stop growing once the window fills, so doubling
    /// context must cost far less than double.
    #[test]
    fn swa_kv_growth_flattens_past_the_window() {
        let g = ModelSpec {
            id: "gemma-3-12b".into(),
            n_layers: 48,
            n_embd: 3840,
            n_vocab: 262144,
            params: 12_200_000_000,
            attention: Attention::Swa {
                n_kv_heads: 8,
                head_dim: 256,
                window: 1024,
                global_every: 6,
            },
            moe: None,
            n_ctx_train: None,
            quants: vec![],
        };
        let at_8k = g.kv_bytes(8192, KvPrecision::F16) as f64;
        let at_16k = g.kv_bytes(16384, KvPrecision::F16) as f64;

        // Doubling context must cost clearly less than double, because the
        // 40 local layers are pinned at the 1024-token window and only the
        // 8 global layers grow. Hand-computed: (8*16384 + 40*1024) /
        // (8*8192 + 40*1024) = 172032 / 106496 = 1.6154.
        let ratio = at_16k / at_8k;
        assert!((1.0..1.7).contains(&ratio), "ratio {ratio}");

        // The stronger property: at 8K, SWA holds ~27% of what full attention
        // over all 48 layers would need. This is the number that decides
        // whether the model fits, so assert it directly.
        let full_attention = ModelSpec {
            attention: Attention::Gqa { n_kv_heads: 8, head_dim: 256 },
            ..g.clone()
        };
        let saving = at_8k / full_attention.kv_bytes(8192, KvPrecision::F16) as f64;
        assert!(saving < 0.30, "SWA should save >70% at 8K, saved {saving}");
    }

    /// SSM layers hold constant state, so a hybrid model's KV grows only with
    /// its attention layers.
    #[test]
    fn hybrid_kv_grows_only_with_attention_layers() {
        let h = ModelSpec {
            id: "hybrid-test".into(),
            n_layers: 64,
            n_embd: 4096,
            n_vocab: 128000,
            params: 8_000_000_000,
            attention: Attention::Hybrid {
                attn_layers: 8,
                n_kv_heads: 8,
                head_dim: 128,
                ssm_state_bytes: 512 * 1024,
            },
            moe: None,
            n_ctx_train: None,
            quants: vec![],
        };
        let full_attn = llama3_8b().kv_bytes(32768, KvPrecision::F16);
        // 8 attention layers vs 32 -> a quarter of the growth.
        assert!(h.kv_bytes(32768, KvPrecision::F16) < full_attn / 3);
    }

    /// MoE reads only the shared core plus top-k experts per token.
    #[test]
    fn moe_active_params_are_a_small_fraction() {
        let m = ModelSpec {
            id: "qwen3-30b-a3b".into(),
            n_layers: 48,
            n_embd: 2048,
            n_vocab: 151936,
            params: 30_500_000_000,
            attention: Attention::Gqa { n_kv_heads: 4, head_dim: 128 },
            moe: Some(Moe {
                n_expert: 128,
                n_active: 8,
                expert_params: 25_000_000,
                shared_params: 2_400_000_000,
            }),
            n_ctx_train: None,
            quants: vec![],
        };
        let frac = m.active_params() as f64 / m.params as f64;
        assert!(frac > 0.05 && frac < 0.45, "active fraction {frac}");
    }

    /// Calibration evidence must actually move the prediction and narrow the
    /// published range. Without this the whole dataset is decorative.
    #[test]
    fn calibration_shifts_eta_and_narrows_the_range() {
        use crate::fit::{Fit, Record};
        let spec = llama3_8b();
        let quant = Quant {
            name: "Q4_K_M".into(),
            bytes: 4_920_733_696,
            family: crate::spec::QuantFamily::KQuant,
        };
        let hw = Hardware {
            backend: Backend::Metal,
            ram_bw_gbs: 132.0,
            ram_bw_peak_gbs: 132.0,
            disk_gbs: 5.0,
            gflops: 420.0,
            usable_bytes: 12 << 30,
            threads: 4,
            vram_bytes: 0,
            vram_bw_gbs: 0.0,
            vram_bw_measured: false,
        };

        let before = predict(&spec, &quant, &hw, KvPrecision::Q8, 2048, 512);
        assert_eq!(before.confidence, Confidence::Prior);
        // No prefill measurement anywhere, so TTFT must be unknown, not wrong.
        assert!(before.prefill_tok_s.is_none());
        assert!(before.ttft_s.is_none());

        // 40 consistent runs saying the machine really achieves eta = 0.85.
        let records: Vec<Record> = (0..40)
            .map(|i| Record {
                hw: format!("hw{i}"),
                backend: "Metal".into(),
                model: "m".into(),
                quant: "Q4_K_M".into(),
                virt: Some("none".into()),
                quant_family: crate::spec::QuantFamily::KQuant,
                implied_eta: 0.85,
                error_pct: None,
                within_range: None,
                implied_prefill_scale: None,
            })
            .collect();
        let after = predict_with(
            &spec, &quant, &hw, KvPrecision::Q8, 2048, 512,
            &Fit::from_records(&records),
        );

        assert_eq!(after.confidence, Confidence::High);
        assert!((after.assumed_eta - 0.85).abs() < 1e-9);
        // Prior was 0.70 * 0.88 = 0.616, so the prediction must rise.
        let mid = |p: &Prediction| (p.decode_tok_s.0 + p.decode_tok_s.1) / 2.0;
        assert!(mid(&after) > mid(&before) * 1.3, "{} vs {}", mid(&after), mid(&before));
        // And the range must be tighter than the uncalibrated one.
        let width = |p: &Prediction| (p.decode_tok_s.1 - p.decode_tok_s.0) / mid(p);
        assert!(width(&after) < width(&before), "{} vs {}", width(&after), width(&before));
    }

    fn hw_16gb() -> Hardware {
        Hardware {
            backend: Backend::Metal,
            ram_bw_gbs: 132.0,
            ram_bw_peak_gbs: 132.0,
            disk_gbs: 5.0,
            gflops: 420.0,
            usable_bytes: 12 << 30,
            threads: 4,
            vram_bytes: 0,
            vram_bw_gbs: 0.0,
            vram_bw_measured: false,
        }
    }

    fn q(name: &str, bytes: u64, family: crate::spec::QuantFamily) -> Quant {
        Quant { name: name.into(), bytes, family }
    }

    /// The inversion must agree with the model it inverts.
    ///
    /// This is the whole safety of `zc plan`: predict a decode rate at a known
    /// bandwidth, feed that rate back through `required_bandwidth_gbs`, and
    /// the bandwidth must come back. A plausible-looking inversion that
    /// disagreed with `predict` would be a confidently wrong number recommending
    /// hardware, which is the worst thing this project could print.
    #[test]
    fn required_bandwidth_round_trips_through_predict() {
        let spec = llama3_70b();
        // Small enough to be fully resident in the 12 GiB budget, so bandwidth
        // is the only term -- the regime the inversion is exact in.
        let quant = q("Q4_K_M", 6 << 30, crate::spec::QuantFamily::KQuant);
        for bw in [30.0f64, 60.0, 132.0, 400.0, 1000.0] {
            let mut hw = hw_16gb();
            hw.ram_bw_gbs = bw;
            let p = predict(&spec, &quant, &hw, KvPrecision::F16, 2048, 512);
            assert!(
                (p.resident_fraction - 1.0).abs() < 1e-9,
                "fixture must be fully resident, got {}",
                p.resident_fraction
            );
            // Midpoint of the published range is eta / seconds_per_token.
            let tps = p.assumed_eta / p.raw_seconds_per_token;
            let back = required_bandwidth_gbs(&spec, &quant, tps, p.assumed_eta);
            assert!(
                (back - bw).abs() / bw < 1e-6,
                "asked for {tps:.3} t/s, inversion said {back:.3} GB/s, machine had {bw}"
            );
        }
    }

    /// Doubling the target doubles the bandwidth, and a bigger active set costs
    /// proportionally more. Both follow from decode being memory-bound.
    #[test]
    fn required_bandwidth_scales_with_target_and_active_bytes() {
        let spec = llama3_70b();
        let small = q("Q4_K_M", 4 << 30, crate::spec::QuantFamily::KQuant);
        let big = q("Q8_0", 8 << 30, crate::spec::QuantFamily::Legacy);
        let a = required_bandwidth_gbs(&spec, &small, 10.0, 0.8);
        assert!((required_bandwidth_gbs(&spec, &small, 20.0, 0.8) - 2.0 * a).abs() < 1e-9);
        assert!((required_bandwidth_gbs(&spec, &big, 10.0, 0.8) - 2.0 * a).abs() < 1e-9);
        // A nonsense target is zero, not a division by zero.
        assert_eq!(required_bandwidth_gbs(&spec, &small, 0.0, 0.8), 0.0);
        assert_eq!(required_bandwidth_gbs(&spec, &small, 10.0, 0.0), 0.0);
    }

    /// The memory a plan reports must be the memory a prediction charges, or
    /// the two surfaces disagree about the same model.
    #[test]
    fn requirement_uses_the_same_terms_as_prediction() {
        let spec = llama3_70b();
        let quant = q("Q4_K_M", 6 << 30, crate::spec::QuantFamily::KQuant);
        let r = requirement(&spec, &quant, 32768, KvPrecision::F16, 512);
        assert_eq!(r.weights, 6 << 30);
        assert_eq!(r.kv, spec.kv_bytes(32768, KvPrecision::F16));
        assert_eq!(r.compute, spec.compute_buffer_bytes(512));
        assert_eq!(r.total, r.weights + r.kv + r.compute);
        // Halving KV precision halves the KV term and nothing else.
        let q4 = requirement(&spec, &quant, 32768, KvPrecision::Q4, 512);
        assert_eq!(q4.weights, r.weights);
        assert!(q4.kv < r.kv, "q4 KV must be smaller than f16");
    }

    /// A model larger than RAM must still run. It streams from disk, slowly.
    /// Reporting WontFit contradicts the streaming model this whole engine is
    /// built around, and used to happen for every large model.
    #[test]
    fn model_larger_than_ram_streams_rather_than_refusing() {
        let spec = llama3_70b();
        // 42 GiB of weights against a 12 GiB budget.
        let quant = q("Q4_K_M", 45_635_198_976, crate::spec::QuantFamily::KQuant);
        let p = predict(&spec, &quant, &hw_16gb(), KvPrecision::Q8, 2048, 512);

        assert_ne!(p.verdict, Verdict::WontFit, "must not refuse a streamable model");
        assert!(p.max_context >= 2048, "context {} too small", p.max_context);
        assert!(p.decode_tok_s.0 > 0.0, "must predict a real, slow speed");
        // Almost nothing fits, so almost everything streams.
        assert!(p.resident_fraction < 0.4, "resident {}", p.resident_fraction);
    }

    /// WontFit is reserved for the genuine case: not even a minimal context
    /// fits alongside the compute buffers.
    #[test]
    fn wont_fit_only_when_context_is_truly_impossible() {
        let spec = llama3_70b();
        let quant = q("Q4_K_M", 45_635_198_976, crate::spec::QuantFamily::KQuant);
        let tiny = Hardware { usable_bytes: 200 << 20, ..hw_16gb() };
        let p = predict(&spec, &quant, &tiny, KvPrecision::Q8, 2048, 512);
        assert_eq!(p.verdict, Verdict::WontFit);
    }

    /// MoE residency must beat the naive blended fraction. The hot core is a
    /// small slice of the file but a large slice of what is read per token, and
    /// it is pinned first — so active bytes are far more resident than a
    /// whole-file ratio suggests.
    #[test]
    fn moe_hot_core_is_pinned_before_experts() {
        let spec = ModelSpec {
            id: "moe".into(),
            n_layers: 48,
            n_embd: 2048,
            n_vocab: 151936,
            params: 30_500_000_000,
            attention: Attention::Gqa { n_kv_heads: 4, head_dim: 128 },
            moe: Some(Moe {
                n_expert: 128,
                n_active: 8,
                expert_params: 4_718_592,
                shared_params: 1_510_000_000,
            }),
            n_ctx_train: None,
            quants: vec![],
        };
        // 19 GiB of weights, 12 GiB budget -> roughly 60% of the file cached.
        let quant = q("Q4_K_M", 19_971_519_744, crate::spec::QuantFamily::KQuant);
        let p = predict(&spec, &quant, &hw_16gb(), KvPrecision::Q8, 2048, 512);

        let whole_file_ratio = 12.0 / 19.0;
        assert!(
            p.resident_fraction > whole_file_ratio,
            "active-byte residency {} should exceed whole-file ratio {whole_file_ratio}",
            p.resident_fraction
        );
    }

    /// The hot core must never be counted as resident when it does not fit.
    /// The old code used `.max(hot)`, which claimed it was.
    #[test]
    fn hot_core_that_does_not_fit_is_not_counted_resident() {
        let spec = ModelSpec {
            id: "moe-big-core".into(),
            n_layers: 48,
            n_embd: 2048,
            n_vocab: 151936,
            params: 30_500_000_000,
            attention: Attention::Gqa { n_kv_heads: 4, head_dim: 128 },
            moe: Some(Moe {
                n_expert: 128,
                n_active: 8,
                expert_params: 4_718_592,
                // Hot core is over half the model.
                shared_params: 16_000_000_000,
            }),
            n_ctx_train: None,
            quants: vec![],
        };
        let quant = q("Q4_K_M", 19_971_519_744, crate::spec::QuantFamily::KQuant);
        let tiny = Hardware { usable_bytes: 2 << 30, ..hw_16gb() };
        let p = predict(&spec, &quant, &tiny, KvPrecision::Q8, 2048, 512);
        assert!(p.resident_fraction < 0.5, "resident {}", p.resident_fraction);
    }

    /// Dense models must degenerate to the simple ratio.
    #[test]
    fn dense_residency_is_the_cache_ratio() {
        let spec = llama3_8b();
        let quant = q("Q4_K_M", 8 << 30, crate::spec::QuantFamily::KQuant);
        let hw = Hardware { usable_bytes: 6 << 30, ..hw_16gb() };
        let p = predict(&spec, &quant, &hw, KvPrecision::Q8, 2048, 512);
        // ~5 GiB cacheable after buffers and the KV floor, out of 8 GiB.
        assert!((0.5..0.75).contains(&p.resident_fraction), "{}", p.resident_fraction);
    }

    fn hw_with_gpu(vram: u64, host: u64) -> Hardware {
        Hardware {
            backend: Backend::Discrete,
            ram_bw_gbs: 50.0,
            ram_bw_peak_gbs: 50.0,
            disk_gbs: 1.0,
            gflops: 420.0,
            usable_bytes: host,
            threads: 8,
            vram_bytes: vram,
            vram_bw_gbs: 360.0,
            vram_bw_measured: false,
        }
    }

    /// A model that fits entirely in VRAM must be read entirely at VRAM
    /// bandwidth. Hand-computed: 4 GiB of active weights over 360 GB/s is
    /// 4294967296 / 3.6e11 = 11.93 ms per token, and nothing else contributes.
    ///
    /// This is the whole point of the tier: before it existed, this machine was
    /// predicted at 50 GB/s — an order of magnitude slow.
    #[test]
    fn weights_that_fit_in_vram_are_read_at_vram_bandwidth() {
        let spec = llama3_8b();
        let quant = q("Q4_K_M", 4 << 30, crate::spec::QuantFamily::KQuant);
        let p = predict(
            &spec,
            &quant,
            &hw_with_gpu(12 << 30, 8 << 30),
            KvPrecision::Q8,
            2048,
            512,
        );

        assert_eq!(p.resident_fraction, 1.0);
        let expected = (4u64 << 30) as f64 / 360e9;
        assert!(
            (p.raw_seconds_per_token - expected).abs() < 1e-12,
            "{} vs {expected}",
            p.raw_seconds_per_token
        );
    }

    /// VRAM is capacity *and* speed, and the two must be modelled separately.
    ///
    /// Same total pool, once as 20 GiB of host RAM and once as 12 GiB of VRAM
    /// plus 8 GiB of host RAM. Capacity being equal, the context available must
    /// be identical; speed being tiered, the GPU machine must be faster — but
    /// only partly, because 16 GiB of weights cannot all fit in 12 GiB of VRAM.
    /// Predicting the full 7.2x speedup would be claiming the spill is free.
    #[test]
    fn a_partial_offload_lands_between_the_two_bandwidths() {
        let spec = llama3_8b();
        let quant = q("Q4_K_M", 16 << 30, crate::spec::QuantFamily::KQuant);
        let cpu_only = Hardware {
            backend: Backend::Cpu,
            ..hw_with_gpu(0, 20 << 30)
        };
        let with_gpu = hw_with_gpu(12 << 30, 8 << 30);

        let a = predict(&spec, &quant, &cpu_only, KvPrecision::Q8, 2048, 512);
        let b = predict(&spec, &quant, &with_gpu, KvPrecision::Q8, 2048, 512);

        assert_eq!(a.max_context, b.max_context, "same pool, same context");
        let speedup = a.raw_seconds_per_token / b.raw_seconds_per_token;
        assert!(speedup > 1.5, "GPU must help: {speedup}");
        assert!(speedup < 360.0 / 50.0, "spill is not free: {speedup}");
    }

    /// Everything else in `Hardware` is measured on the machine. VRAM
    /// bandwidth is a table lookup, so a prediction resting on it must not be
    /// able to reach `high` no matter how many calibration runs back the
    /// coefficient — those runs came from other machines with other cards.
    #[test]
    fn a_looked_up_vram_bandwidth_cannot_earn_high_confidence() {
        use crate::fit::{Fit, Record};
        let spec = llama3_8b();
        let quant = q("Q4_K_M", 4 << 30, crate::spec::QuantFamily::KQuant);
        let records: Vec<Record> = (0..40)
            .map(|i| Record {
                hw: format!("hw{i}"),
                backend: "Discrete".into(),
                model: "m".into(),
                quant: "Q4_K_M".into(),
                virt: Some("none".into()),
                quant_family: crate::spec::QuantFamily::KQuant,
                implied_eta: 0.85,
                error_pct: None,
                within_range: None,
                implied_prefill_scale: None,
            })
            .collect();
        let fit = Fit::from_records(&records);
        let hw = hw_with_gpu(12 << 30, 8 << 30);

        let p = predict_with(&spec, &quant, &hw, KvPrecision::Q8, 2048, 512, &fit);
        assert_eq!(p.confidence, Confidence::Low);
        let width = (p.decode_tok_s.1 - p.decode_tok_s.0) / (p.decode_tok_s.1 + p.decode_tok_s.0);
        assert!(width >= 0.25, "range must stay wide: {width}");

        // Measure the same bandwidth and the cap lifts — the cap is about
        // provenance, not about GPUs.
        let measured = Hardware { vram_bw_measured: true, ..hw };
        let m = predict_with(&spec, &quant, &measured, KvPrecision::Q8, 2048, 512, &fit);
        assert_eq!(m.confidence, Confidence::High);
    }

    /// Two machines in one bucket that disagree by 8x produce a MAD spread
    /// wider than 100%, and `eta * (1 - spread)` then goes negative. This is
    /// real: `gate.jsonl` held exactly such a pair, and `zc check --all`
    /// printed `-0.2-0.7` tok/s for the models it hit. A rate below zero is
    /// not a wide prediction, it is a broken one.
    #[test]
    fn a_disagreeing_bucket_never_predicts_a_negative_rate() {
        use crate::fit::{Fit, Record};
        let spec = llama3_8b();
        let quant = q("Q4_K_M", 4 << 30, crate::spec::QuantFamily::KQuant);
        let rec = |eta: f64| Record {
            hw: "x".into(),
            backend: "Cpu".into(),
            model: "m".into(),
            quant: "Q4_K_M".into(),
            virt: Some("none".into()),
            quant_family: crate::spec::QuantFamily::KQuant,
            implied_eta: eta,
            error_pct: None,
            within_range: None,
            implied_prefill_scale: None,
        };
        // 8x apart, the spread seen on the real macOS-VM pair.
        let fit = Fit::from_records(&[rec(0.11), rec(0.92), rec(0.11), rec(0.92)]);
        let p = predict_with(&spec, &quant, &hw_16gb(), KvPrecision::Q8, 2048, 512, &fit);

        assert!(p.decode_tok_s.0 >= 0.0, "low bound {} is negative", p.decode_tok_s.0);
        assert!(p.decode_tok_s.1 > p.decode_tok_s.0, "range must not invert");
    }

    /// max_context must be monotonic in budget, and zero when the weights do
    /// not even fit.
    #[test]
    fn max_context_behaves() {
        let m = llama3_8b();
        let weights = 4_900_000_000;
        assert_eq!(m.max_context(3 << 30, weights, KvPrecision::F16, 512), 0);
        let small = m.max_context(6 << 30, weights, KvPrecision::F16, 512);
        let large = m.max_context(12 << 30, weights, KvPrecision::F16, 512);
        assert!(small > 0 && large > small, "{small} -> {large}");
        // Q4 KV must buy roughly 3.5x the context of F16.
        let q4 = m.max_context(12 << 30, weights, KvPrecision::Q4, 512);
        assert!(q4 as f64 / large as f64 > 3.0, "q4 {q4} vs f16 {large}");
    }
}

/// What a model needs to run at a given context, however that memory is
/// provided — host RAM, VRAM, or unified.
///
/// The inverse of the question `zc check` answers. `check` asks what this
/// machine can run; this asks what running a given thing would take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Requirement {
    pub weights: u64,
    pub kv: u64,
    pub compute: u64,
    pub total: u64,
}

/// Memory needed for `spec` at `quant` and `ctx` tokens.
///
/// Every term comes from the same functions `predict` uses, so a plan and a
/// prediction cannot disagree about how much memory a model wants.
pub fn requirement(
    spec: &ModelSpec,
    quant: &Quant,
    ctx: u32,
    kv_prec: KvPrecision,
    ubatch: u32,
) -> Requirement {
    let weights = quant.bytes;
    let kv = spec.kv_bytes(ctx, kv_prec);
    let compute = spec.compute_buffer_bytes(ubatch);
    Requirement {
        weights,
        kv,
        compute,
        total: weights.saturating_add(kv).saturating_add(compute),
    }
}

/// Memory bandwidth needed to decode at `target_tps`, in GB/s.
///
/// Decode is memory-bound: every token reads the active weight set once, so
/// `decode = eta * bandwidth / active_bytes` and this is that solved for
/// bandwidth. Exact whenever the weights are resident, which is the regime a
/// plan is about — if they are not resident the binding constraint is memory,
/// not bandwidth, and [`requirement`] is the number that matters.
///
/// Stated in GB/s rather than as a GPU model name on purpose. A name is a
/// lookup, and a lookup is the thing this project refuses to put underneath a
/// number; bandwidth is the unit that actually governs decode and it is
/// checkable against any spec sheet.
pub fn required_bandwidth_gbs(spec: &ModelSpec, quant: &Quant, target_tps: f64, eta: f64) -> f64 {
    if target_tps <= 0.0 || eta <= 0.0 {
        return 0.0;
    }
    spec.active_bytes(quant) as f64 * target_tps / eta / 1e9
}

/// The efficiency a plan should assume for this backend and quantisation.
///
/// Same lookup `predict` performs, exposed so a plan states the coefficient it
/// used and where the evidence came from rather than hiding it.
pub fn plan_eta(fit: &Fit, backend: Backend, quant: &Quant) -> crate::fit::Coefficient {
    let prior = DECODE_EFFICIENCY * quant.family.efficiency();
    fit.lookup(&format!("{backend:?}"), quant.family, prior)
}
