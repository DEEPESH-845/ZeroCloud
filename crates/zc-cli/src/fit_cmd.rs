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
    print!("{}", summary_text(&path()));
    0
}

/// The body of `zc fit`, as text.
///
/// Returned rather than printed so `zc doctor` can embed the identical block.
/// A bug report that shows different coefficients from the ones the user sees
/// would send everyone chasing a discrepancy that is not there.
pub fn summary_text(path: &std::path::Path) -> String {
    use std::fmt::Write;
    let fit = Fit::load(path);
    let mut o = String::new();

    if fit.is_empty() {
        let _ = writeln!(o, "No calibration data at {}.\n", path.display());
        let _ = writeln!(o, "Every tok/s number zc reports is currently a shipped prior.");
        let _ = writeln!(o, "To replace priors with measurements:");
        let _ = writeln!(o, "    ollama pull qwen3:4b && zc verify");
        return o;
    }

    let _ = writeln!(o, "== fitted coefficients ==  ({})\n", path.display());
    let _ = writeln!(
        o,
        "  {:<24} {:>7} {:>9} {:>9}  confidence",
        "bucket", "runs", "eta", "spread"
    );
    for (key, c) in fit.summary() {
        let _ = writeln!(
            o,
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
        let _ = writeln!(o, "\n  Thin evidence in: {}", thin.join(", "));
        let _ = writeln!(o, "  8 runs per bucket reaches medium confidence, 30 reaches high.");
    }
    o
}
