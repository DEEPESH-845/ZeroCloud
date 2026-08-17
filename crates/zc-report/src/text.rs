//! Human-readable rendering. This is the output `zc check` has always had.

use crate::Report;
use zc_model::{Backend, Verdict};
use zc_probe::human;

pub fn render(r: &Report) -> String {
    let mut o = String::with_capacity(4096);
    let p = &mut o;

    push(p, "== hardware ==");
    push(
        p,
        &format!(
            "  {}   {}P+{}E   {} total / {} available{}",
            r.cpu.brand,
            r.cpu.p_cores,
            r.cpu.e_cores,
            human(r.mem.total),
            human(r.mem.available),
            if r.mem.unified { "   unified" } else { "" }
        ),
    );
    push(
        p,
        &format!(
            "  {} on {} ({:?}){}",
            r.storage.mount.display(),
            r.storage.fstype,
            r.storage.medium,
            if r.storage.bench_file.is_some() && !r.storage.bench_is_weight {
                "   [benchmarked against a non-model file on the same volume]"
            } else {
                ""
            }
        ),
    );

    for g in r.gpus {
        let count = if g.count > 1 {
            format!(" x{}", g.count)
        } else {
            String::new()
        };
        push(
            p,
            &format!(
                "  {}{}   {}",
                g.name,
                count,
                if g.integrated {
                    // Says nothing about VRAM on purpose: an integrated GPU has
                    // none of its own, and Windows' "shared system memory"
                    // figure is memory it takes from the CPU, not memory it adds.
                    "integrated (shares system memory)".to_string()
                } else {
                    format!(
                        "{} VRAM   ~{:.0} GB/s (looked up, not measured)",
                        human(g.vram_bytes),
                        g.bw_gbs
                    )
                }
            ),
        );
    }

    push(p, "\n== measured ==");
    let mut curve = String::new();
    for (i, (t, g)) in r.ram.by_threads.iter().enumerate() {
        if i > 0 {
            curve.push_str("  ");
        }
        curve.push_str(&format!("{t}t:{g:.0}"));
    }
    push(
        p,
        &format!(
            "  ram          {:.0} GB/s peak @{}t   [{curve}]",
            r.ram.peak_gbs, r.ram.peak_threads
        ),
    );
    push(
        p,
        &format!(
            "  compute      {:.0} GFLOPS f32 @{}t   {:.0} GOPS int8 ({:.2}x)",
            r.compute.gflops_nt, r.compute.threads, r.compute.int8_gops_nt, r.compute.int8_ratio
        ),
    );
    match r.disk {
        Some(d) => push(
            p,
            &format!(
                "  disk         {:.2} GB/s random 128K @QD{}   {:.0}K IOPS 4K{}",
                d.rand_128k_qdn_gbs,
                d.queue_depth,
                d.rand_4k_qdn_iops / 1000.0,
                if d.uncached { "" } else { "   [CACHED - inflated]" }
            ),
        ),
        // A missing disk measurement is load-bearing: streaming speed is
        // predicted from it, so silently substituting a default would produce
        // confident nonsense for every over-budget model.
        None => push(
            p,
            "  disk         MEASUREMENT FAILED - streaming predictions unreliable",
        ),
    }
    push(
        p,
        &format!(
            "  budget       {} on an idle machine   ({} free right now)",
            human(r.budget_idle),
            human(r.budget_now)
        ),
    );

    let a = &r.assumptions;
    // On a discrete GPU the weights that matter sit in VRAM, so quoting the
    // host RAM figure next to a GPU-speed prediction would read as a
    // contradiction.
    let bw = if r.backend == Backend::Discrete {
        format!("{:.0} GB/s VRAM + {:.0} GB/s RAM", r.vram_bw_gbs, r.ram_bw_gbs)
    } else {
        format!("{:.0} GB/s", r.ram_bw_gbs)
    };
    push(
        p,
        &format!(
            "\n== predictions ==  ({:?} backend, {bw}, KV at {}, {}-token prompt)",
            r.backend,
            a.kv_precision.to_uppercase(),
            a.prompt_tokens
        ),
    );
    if r.backend == Backend::Discrete {
        push(
            p,
            "  VRAM bandwidth is a table lookup, not a measurement - confidence is capped at 'low'",
        );
    }
    if a.idle_machine {
        push(p, "  assumes an otherwise-idle machine");
    }
    if a.uncalibrated {
        push(
            p,
            "  no calibration data yet - ranges are wide priors. Run `zc verify`.",
        );
    }
    push(
        p,
        &format!(
            "  {:<30} {:<8} {:>13} {:>9} {:>7}  {:<6} verdict",
            "model", "quant", "decode tok/s", "max ctx", "TTFT", "conf"
        ),
    );

    for row in &r.models {
        let pr = &row.prediction;
        let mark = match pr.verdict {
            Verdict::Good => "OK ",
            Verdict::Usable => "ok ",
            Verdict::Slow => "SLOW",
            Verdict::WontFit => "XX ",
        };
        let ctx = if pr.max_context == 0 {
            "-".to_string()
        } else if pr.max_context >= 1024 {
            format!("{}K", pr.max_context / 1024)
        } else {
            pr.max_context.to_string()
        };
        let speed = if pr.verdict == Verdict::WontFit {
            "-".to_string()
        } else {
            format!("{:.1}-{:.1}", pr.decode_tok_s.0, pr.decode_tok_s.1)
        };
        // TTFT is unknown until a real run has been measured on this backend.
        // A dash is honest; a derived number would be wrong by an unknown
        // factor (see zc-model::fit::prefill_scale).
        let ttft = match pr.ttft_s {
            Some(t) if t < 100.0 => format!("{t:.1}s"),
            Some(t) => format!("{t:.0}s"),
            None => "-".to_string(),
        };
        push(
            p,
            &format!(
                "  {:<30} {:<8} {:>13} {:>9} {:>7}  {:<6} {} {}",
                row.model_id,
                row.quant.name,
                speed,
                ctx,
                ttft,
                pr.confidence.label(),
                mark,
                if pr.resident_fraction < 0.999 {
                    format!("{:.0}% resident", pr.resident_fraction * 100.0)
                } else {
                    String::new()
                }
            ),
        );
    }

    if a.prefill_unmeasured {
        push(
            p,
            "\n  TTFT shown as '-': prefill speed cannot be derived from a CPU benchmark",
        );
        push(
            p,
            "  (real prefill runs on the GPU or via int8 kernels). Run `zc verify` to measure it.",
        );
    }

    if r.models.len() < a.total_rows {
        push(
            p,
            &format!(
                "\n  showing {} of {} - ranked by verdict, then speed, then context (--all for every quant)",
                r.models.len(),
                a.total_rows
            ),
        );
    }

    for h in &r.storage.hazards {
        push(p, &format!("\n  !  {h:?}"));
    }
    if let Some(w) = &r.env.warning {
        push(p, &format!("\n  !  {w}"));
    }

    o
}

fn push(out: &mut String, line: &str) {
    out.push_str(line);
    out.push('\n');
}
