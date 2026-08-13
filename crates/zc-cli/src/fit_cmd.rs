//! `zc fit` — what the calibration dataset currently says.
//!
//! Makes the evidence behind every prediction inspectable. A user who wants to
//! know why we claim 20 tok/s can see exactly how many real runs that rests on.

use zc_model::{fit::Confidence, Fit};

const DEFAULT_PATH: &str = "data/calibration/local.jsonl";

/// Overridable so tests and validation runs cannot contaminate a real dataset.
pub fn path() -> std::path::PathBuf {
    std::env::var("ZC_CALIBRATION")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from(DEFAULT_PATH))
}

pub fn run() -> i32 {
    let path = path();
    let fit = Fit::load(&path);

    if fit.is_empty() {
        println!("No calibration data at {}.\n", path.display());
        println!("Every tok/s number zc reports is currently a shipped prior.");
        println!("To replace priors with measurements:");
        println!("    ollama pull qwen3:4b && zc verify");
        return 0;
    }

    println!("== fitted coefficients ==  ({})\n", path.display());
    println!("  {:<24} {:>7} {:>9} {:>9}  confidence", "bucket", "runs", "eta", "spread");
    for (key, c) in fit.summary() {
        println!(
            "  {:<24} {:>7} {:>9.3} {:>8.0}%  {}",
            key,
            c.samples,
            c.eta,
            c.spread * 100.0,
            c.confidence.label()
        );
    }

    let thin: Vec<_> = fit
        .summary()
        .into_iter()
        .filter(|(_, c)| matches!(c.confidence, Confidence::Low))
        .map(|(k, _)| k)
        .collect();
    if !thin.is_empty() {
        println!("\n  Thin evidence in: {}", thin.join(", "));
        println!("  8 runs per bucket reaches medium confidence, 30 reaches high.");
    }
    0
}
