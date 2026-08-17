//! KV cache sizing.
//!
//! This is the number that actually kills people. Not "the model doesn't fit"
//! — that fails at load, loudly. KV runs out at message twelve, after the
//! conversation is already going, and looks like a crash.
//!
//! A 70B model at 32K context needs ~10 GiB of KV on its own, more than the
//! whole machine we are targeting. Nothing else in capacity planning has this
//! much leverage, and no consumer tool surfaces it.

use crate::spec::{Attention, ModelSpec};

/// Bits per KV element. Quantising the cache is the single highest-leverage
/// setting on a memory-constrained machine: Q8 halves it, Q4 quarters it, for
/// very little quality cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPrecision {
    F16,
    Q8,
    Q4,
}

impl KvPrecision {
    /// What a runtime uses unless told otherwise.
    ///
    /// Every runtime we speak to ships f16 KV by default — Ollama needs
    /// `OLLAMA_KV_CACHE_TYPE` *and* flash attention to change it, llama.cpp
    /// needs `--cache-type-k`. Predicting Q8 while the user runs f16 nearly
    /// doubles the context we advertise, which is the one number people decide
    /// on. None of the three reports its KV type over its API, so this is a
    /// stated assumption the user can override, not a reading.
    pub const DEFAULT: Self = KvPrecision::F16;

    /// Parse a CLI value. `None` for anything unrecognised, so a typo becomes
    /// an error rather than a silent fallback to a different memory model.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f16" | "fp16" => Some(KvPrecision::F16),
            "q8" | "q8_0" => Some(KvPrecision::Q8),
            "q4" | "q4_0" => Some(KvPrecision::Q4),
            _ => None,
        }
    }

    /// Stable tag for reports and records.
    pub fn tag(self) -> &'static str {
        match self {
            KvPrecision::F16 => "f16",
            KvPrecision::Q8 => "q8",
            KvPrecision::Q4 => "q4",
        }
    }

    /// Bytes per element, including quantisation scale overhead.
    pub fn bytes(self) -> f64 {
        match self {
            KvPrecision::F16 => 2.0,
            // Q8_0 packs 32 values + one f16 scale per block: 34/32 bytes.
            KvPrecision::Q8 => 34.0 / 32.0,
            // Q4_0 packs 32 values in 16 bytes + one f16 scale: 18/32 bytes.
            KvPrecision::Q4 => 18.0 / 32.0,
        }
    }
}

impl ModelSpec {
    /// Total KV cache bytes at a given context length.
    ///
    /// Not simply `per_token * ctx` — sliding-window layers stop growing once
    /// the window fills, and SSM layers never grow at all. Those shapes are
    /// why this takes `ctx` rather than returning a per-token rate.
    pub fn kv_bytes(&self, ctx: u32, prec: KvPrecision) -> u64 {
        let e = prec.bytes();
        let l = self.n_layers as f64;

        let bytes = match &self.attention {
            // K and V stored separately -> factor 2.
            Attention::Gqa { n_kv_heads, head_dim } => {
                2.0 * l * *n_kv_heads as f64 * *head_dim as f64 * ctx as f64 * e
            }

            // One compressed latent per layer; no separate K and V, so no
            // factor of 2. Width is the LoRA rank plus the un-compressed
            // rotary part.
            Attention::Mla { kv_lora_rank, rope_head_dim } => {
                l * (*kv_lora_rank + *rope_head_dim) as f64 * ctx as f64 * e
            }

            // Local layers cap at `window`; global layers grow with ctx.
            Attention::Swa { n_kv_heads, head_dim, window, global_every } => {
                let per_layer_token = 2.0 * *n_kv_heads as f64 * *head_dim as f64 * e;
                let global = (self.n_layers / global_every.max(&1)) as f64;
                let local = l - global;
                let local_ctx = ctx.min(*window) as f64;
                per_layer_token * (global * ctx as f64 + local * local_ctx)
            }

            // Only attention layers accumulate; SSM layers hold constant state.
            Attention::Hybrid { attn_layers, n_kv_heads, head_dim, ssm_state_bytes } => {
                let ssm_layers = l - *attn_layers as f64;
                2.0 * *attn_layers as f64 * *n_kv_heads as f64 * *head_dim as f64 * ctx as f64 * e
                    + ssm_layers * *ssm_state_bytes as f64
            }
        };
        bytes as u64
    }

    /// Marginal KV cost of one more token at the given context.
    ///
    /// Reported as a rate for the UI. For SWA this legitimately falls to near
    /// zero once the window is full, which is the whole point of SWA.
    pub fn kv_bytes_per_token(&self, ctx: u32, prec: KvPrecision) -> u64 {
        let a = self.kv_bytes(ctx, prec);
        let b = self.kv_bytes(ctx + 256, prec);
        (b.saturating_sub(a)) / 256
    }

    /// Transient buffers llama.cpp allocates for the compute graph.
    ///
    /// Dominated by activation tensors sized `ubatch * n_embd`, plus a small
    /// logits buffer (only the last position by default, so `n_vocab * 4`
    /// rather than `n_vocab * ubatch * 4` — that larger term only appears with
    /// `logits_all`).
    ///
    /// The multiplier is an empirical stand-in for the number of live tensors
    /// in the graph and is a primary calibration target.
    pub fn compute_buffer_bytes(&self, ubatch: u32) -> u64 {
        const LIVE_TENSORS: u64 = 20;
        let activations = ubatch as u64 * self.n_embd as u64 * 4 * LIVE_TENSORS;
        let logits = self.n_vocab as u64 * 4;
        activations + logits + (64 << 20) // fixed overhead: scratch, backend
    }

    /// Largest context whose KV cache fits in `for_kv` bytes.
    ///
    /// Binary search rather than division because SWA and hybrid layers make
    /// the relationship non-linear.
    pub fn max_context_in(&self, for_kv: u64, prec: KvPrecision) -> u32 {
        let (mut lo, mut hi) = (0u32, 1_048_576u32);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if self.kv_bytes(mid, prec) <= for_kv {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }

    /// Largest context that fits in `budget` bytes alongside fully-resident
    /// weights. Used where weights genuinely cannot stream.
    ///
    /// Binary search rather than division because SWA and hybrid layers make
    /// the relationship non-linear.
    pub fn max_context(
        &self,
        budget: u64,
        weight_bytes: u64,
        prec: KvPrecision,
        ubatch: u32,
    ) -> u32 {
        let fixed = weight_bytes + self.compute_buffer_bytes(ubatch);
        if fixed >= budget {
            return 0;
        }
        let for_kv = budget - fixed;

        let (mut lo, mut hi) = (0u32, 1_048_576u32);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if self.kv_bytes(mid, prec) <= for_kv {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }
}
