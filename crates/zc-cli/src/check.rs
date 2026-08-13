//! `zc check` — what can this machine run, and how fast?

use crate::machine::Machine;
use zc_model::{catalog, predict, Fit, KvPrecision, Verdict};
use zc_probe::human;

const UBATCH: u32 = 512;
const PROMPT: u32 = 2048;

pub fn run(m: &Machine, fit: &Fit) -> i32 {
    println!("== hardware ==");
    println!("  {}   {}P+{}E   {} total / {} available{}",
        m.cpu.brand, m.cpu.p_cores, m.cpu.e_cores,
        human(m.mem.total), human(m.mem.available),
        if m.mem.unified { "   unified" } else { "" });
    println!("  {} on {} ({:?}){}", m.storage.mount.display(), m.storage.fstype,
        m.storage.medium,
        if m.storage.bench_file.is_some() && !m.storage.bench_is_weight {
            "   [benchmarked against a non-model file on the same volume]"
        } else { "" });

    println!("\n== measured ==");
    print!("  ram          {:.0} GB/s peak @{}t   [", m.ram.peak_gbs, m.ram.peak_threads);
    for (i, (t, g)) in m.ram.by_threads.iter().enumerate() {
        if i > 0 { print!("  "); }
        print!("{t}t:{g:.0}");
    }
    println!("]");
    println!("  compute      {:.0} GFLOPS f32 @{}t   {:.0} GOPS int8 ({:.2}x)",
        m.compute.gflops_nt, m.compute.threads, m.compute.int8_gops_nt, m.compute.int8_ratio);
    if let Some(d) = &m.disk {
        println!("  disk         {:.2} GB/s random 128K @QD{}   {:.0}K IOPS 4K{}",
            d.rand_128k_qdn_gbs, d.queue_depth, d.rand_4k_qdn_iops / 1000.0,
            if d.uncached { "" } else { "   [CACHED - inflated]" });
    } else {
        // A missing disk measurement is load-bearing: streaming speed is
        // predicted from it, so silently substituting a default would produce
        // confident nonsense for every over-budget model.
        println!("  disk         MEASUREMENT FAILED - streaming predictions unreliable");
    }
    println!("  budget       {} on an idle machine   ({} free right now)",
        human(m.budget_idle), human(m.budget_now));

    println!("\n== predictions ==  ({:?} backend, {:.0} GB/s, KV at Q8, {PROMPT}-token prompt)",
        m.backend, m.hw.ram_bw_gbs);
    println!("  assumes an otherwise-idle machine");
    if fit.is_empty() {
        println!("  no calibration data yet - ranges are wide priors. Run `zc verify`.");
    }
    println!("  {:<22} {:<9} {:>13} {:>10} {:>8}  {:<8} verdict",
        "model", "quant", "decode tok/s", "max ctx", "TTFT", "conf");

    for spec in catalog::load() {
        for quant in &spec.quants {
            let p = predict::predict_with(&spec, quant, &m.hw, KvPrecision::Q8, PROMPT, UBATCH, fit);
            let mark = match p.verdict {
                Verdict::Good => "OK ", Verdict::Usable => "ok ",
                Verdict::Slow => "SLOW", Verdict::WontFit => "XX ",
            };
            let ctx = if p.max_context == 0 { "-".to_string() }
                      else if p.max_context >= 1024 { format!("{}K", p.max_context / 1024) }
                      else { p.max_context.to_string() };
            let speed = if p.verdict == Verdict::WontFit { "-".into() }
                        else { format!("{:.1}-{:.1}", p.decode_tok_s.0, p.decode_tok_s.1) };
            // TTFT is unknown until a real run has been measured on this
            // backend. A dash is honest; a derived number would be wrong by an
            // unknown factor (see zc-model::fit::prefill_scale).
            let ttft = match p.ttft_s {
                Some(t) if t < 100.0 => format!("{t:.1}s"),
                Some(t) => format!("{t:.0}s"),
                None => "-".into(),
            };
            println!("  {:<22} {:<9} {:>13} {:>10} {:>8}  {:<8} {} {}",
                spec.id, quant.name, speed, ctx, ttft, p.confidence.label(), mark,
                if p.resident_fraction < 0.999 {
                    format!("{:.0}% resident", p.resident_fraction * 100.0)
                } else { String::new() });
        }
    }

    if fit.prefill_scale(&format!("{:?}", m.backend)).is_none() {
        println!("\n  TTFT shown as '-': prefill speed cannot be derived from a CPU benchmark");
        println!("  (real prefill runs on the GPU or via int8 kernels). Run `zc verify` to measure it.");
    }

    for h in &m.storage.hazards { println!("\n  !  {h:?}"); }
    if let Some(w) = &m.env.warning { println!("\n  !  {w}"); }
    0
}
