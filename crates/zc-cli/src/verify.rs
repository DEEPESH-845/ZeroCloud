//! `zc verify` — run a real model and find out whether we were right.
//!
//! Everything else in this project produces predictions. This is the only
//! command that produces *facts*, and it is what turns `DECODE_EFFICIENCY`
//! from a guess into a measurement.

use crate::machine::Machine;
use zc_model::{predict, KvPrecision};
use zc_runtime::{calibrate, ollama};

/// Enough tokens for the rate to stabilise past the first-token transient,
/// short enough that a slow machine finishes in about a minute.
const NUM_PREDICT: u32 = 128;
const NUM_CTX: u32 = 4096;
/// Approximate token count of `measurement_prompt`. The runtime reports the
/// exact figure back in `prompt_eval_count`; this is only used to predict TTFT
/// for the same prompt we are about to send.
const PROMPT_TOKENS: u32 = 1200;

/// Confirm a runtime is reachable *before* the caller spends 20 seconds
/// benchmarking. Telling a user to install Ollama after making them wait is a
/// small cruelty that costs nothing to avoid.
pub fn precheck() -> Result<ollama::Endpoint, i32> {
    let ep = ollama::Endpoint::from_env();
    if ep.is_running() {
        return Ok(ep);
    }
    eprintln!("No runtime found at {ep}.\n");
    eprintln!("zc verify needs a real model to measure against. Install Ollama:");
    eprintln!("    brew install ollama && ollama serve");
    eprintln!("    ollama pull qwen3:4b");
    eprintln!("\nThen run `zc verify` again.");
    eprintln!("(Set OLLAMA_HOST if your runtime listens elsewhere.)");
    Err(2)
}

pub fn run(m: &Machine, ep: &ollama::Endpoint, fit: &zc_model::Fit, wanted: Option<&str>) -> i32 {
    let models = match ep.list() {
        Ok(v) if !v.is_empty() => v,
        Ok(_) => {
            eprintln!("Runtime is up but has no models. Try: ollama pull qwen3:4b");
            return 2;
        }
        Err(e) => {
            eprintln!("Could not list models: {e}");
            return 1;
        }
    };

    // Default to the smallest model: it is the fastest to measure and the most
    // likely to fit, and a first run that takes ten minutes is a first run
    // nobody finishes.
    let chosen = match wanted {
        Some(name) => match models.iter().find(|m| m.name.starts_with(name)) {
            Some(m) => m,
            None => {
                eprintln!("No installed model matches '{name}'. Available:");
                for m in &models {
                    eprintln!("    {}  ({:.2} GiB)", m.name, m.size_bytes as f64 / 1e9);
                }
                return 2;
            }
        },
        None => models.last().expect("non-empty"),
    };

    let spec = match ep.describe(chosen) {
        Ok(Some(s)) => s,
        Ok(None) => {
            eprintln!("Runtime did not report usable metadata for {}.", chosen.name);
            return 1;
        }
        Err(e) => {
            eprintln!("Could not describe {}: {e}", chosen.name);
            return 1;
        }
    };
    let quant = &spec.quants[0];

    println!("== verify ==");
    println!("  runtime      {ep}");
    println!(
        "  model        {}   {}   {:.2} GiB",
        chosen.name,
        quant.name,
        chosen.size_bytes as f64 / 1_073_741_824.0
    );
    println!(
        "  spec         {}L  n_embd {}  vocab {}  {}",
        spec.n_layers,
        spec.n_embd,
        spec.n_vocab,
        match &spec.moe {
            Some(mo) => format!(
                "MoE {}x top-{} ({:.2}B active of {:.1}B)",
                mo.n_expert,
                mo.n_active,
                spec.active_params() as f64 / 1e9,
                spec.params as f64 / 1e9
            ),
            None => format!("dense {:.1}B", spec.params as f64 / 1e9),
        }
    );

    let pred = predict::predict_with(
        &spec, quant, &m.hw, KvPrecision::Q8, PROMPT_TOKENS, 512, fit,
    );

    // A cold model would charge load time to the first tokens and understate
    // the rate. Ollama reports load_duration separately, but warming also
    // settles clocks and page cache.
    print!("  warming up   ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(1, |d| d.as_nanos() as u64);
    match ep.generate(&chosen.name, 8, NUM_CTX, nonce) {
        Ok(w) => println!("loaded in {:.1}s", w.load_s),
        Err(e) => {
            println!("failed");
            eprintln!("  {e}");
            return 1;
        }
    }

    print!("  measuring    ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    // A different nonce: Ollama caches KV for repeated prompt prefixes, and a
    // cached prefix would report near-zero prefill time and silently corrupt
    // the calibration.
    let run = match ep.generate(&chosen.name, NUM_PREDICT, NUM_CTX, nonce.wrapping_add(1)) {
        Ok(r) => r,
        Err(e) => {
            println!("failed");
            eprintln!("  {e}");
            return 1;
        }
    };
    println!("{} prompt + {} generated tokens", run.prompt_tokens, run.eval_tokens);

    let cal = calibrate::compare(
        &chosen.name, &quant.name, &pred, &run, spec.active_params(), m.hw.gflops,
    );

    println!("\n  {:<12} {:>18}  {:>16}", "", "predicted", "actual");
    println!(
        "  {:<12} {:>18} {:>15.1}   {}",
        "decode",
        format!("{:.1}-{:.1} tok/s", pred.decode_tok_s.0, pred.decode_tok_s.1),
        cal.actual,
        if cal.within_range {
            "HIT".to_string()
        } else {
            format!("MISS ({:+.0}%)", cal.error_pct)
        }
    );
    println!(
        "  {:<12} {:>18} {:>15.1}   scale {:.2}x vs f32 peak",
        "prefill",
        match pred.prefill_tok_s {
            Some(p) => format!("{p:.0} tok/s"),
            None => "not yet measured".into(),
        },
        cal.actual_prefill_tok_s,
        cal.implied_prefill_scale
    );
    println!(
        "  {:<12} {:>18} {:>15.3}   <- the calibration constant",
        "eta",
        format!("assumed {:.3}", cal.assumed_eta),
        cal.implied_eta
    );

    if pred.resident_fraction < 0.999 {
        println!(
            "\n  note: predicted only {:.0}% resident, so this also exercises the \
             streaming path",
            pred.resident_fraction * 100.0
        );
    }

    let line = calibrate::record_line(
        &calibrate::fingerprint(&m.profile()),
        m.env.os,
        // Collapsed to a fixed vocabulary; see `Env::virt_tag`. Shared with the
        // report so a record and a report can never classify the same machine
        // differently.
        m.env.virt_tag(),
        &format!("{:?}", m.backend),
        m.hw.ram_bw_gbs,
        m.hw.disk_gbs,
        m.hw.gflops,
        m.hw.threads,
        NUM_CTX,
        spec.active_params(),
        &cal,
        &run,
    );
    let path = crate::fit_cmd::path();
    match calibrate::append(&path, &line) {
        Ok(()) => println!("\n  recorded to {}  (local only; `zc share` to submit)", path.display()),
        Err(e) => eprintln!("\n  could not write {}: {e}", path.display()),
    }

    if !cal.within_range {
        println!(
            "\n  A miss is data, not a failure. Run `zc verify` across several models\n  \
             and quant families, then refit eta from data/calibration/local.jsonl."
        );
    }
    0
}
