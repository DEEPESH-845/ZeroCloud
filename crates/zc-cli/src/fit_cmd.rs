//! `zc fit` — what the calibration dataset currently says.
//!
//! Makes the evidence behind every prediction inspectable. A user who wants to
//! know why we claim 20 tok/s can see exactly how many real runs that rests on.

use zc_model::{fit::Confidence, Fit};

const DEFAULT_DIR: &str = "data/calibration";

/// Overridable so tests and validation runs cannot contaminate a real dataset.
pub fn path() -> std::path::PathBuf {
    std::env::var("ZC_CALIBRATION")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| resolve(std::path::Path::new(DEFAULT_DIR)))
}

/// Which file in a calibration directory is the dataset.
///
/// `gate.jsonl` is the curated cross-machine set committed to the repo — it is
/// what the published accuracy figure is computed from, so CI and any reader
/// can recompute that number from a clean checkout. `local.jsonl` is gitignored
/// and holds whatever this machine's `zc verify` produced. Preferring the
/// committed file when it exists is what makes the claim reproducible; falling
/// back keeps a fresh clone with no dataset behaving exactly as before.
///
/// Merging a user's local runs *into* the shipped dataset is `zc share`'s job,
/// not this function's.
fn resolve(dir: &std::path::Path) -> std::path::PathBuf {
    let curated = dir.join("gate.jsonl");
    if curated.is_file() {
        curated
    } else {
        dir.join("local.jsonl")
    }
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

#[cfg(test)]
mod tests {
    use super::resolve;

    /// A unique scratch directory. No dev-dependencies: the workspace's
    /// dependency policy applies to tests too, so no `tempfile` crate.
    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("zc-fit-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A fresh clone ships no curated dataset, so `zc fit` must keep reading
    /// the local file rather than pointing at a path that does not exist.
    /// Without this, every new user's first `zc fit` would report no data.
    #[test]
    fn falls_back_to_local_when_no_curated_dataset() {
        let d = scratch("fallback");
        assert_eq!(resolve(&d), d.join("local.jsonl"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Once the cross-machine dataset is committed it *is* the dataset: it is
    /// what the published accuracy number is computed from, and one machine's
    /// local runs must not silently shadow it.
    #[test]
    fn curated_dataset_wins_when_present() {
        let d = scratch("curated");
        std::fs::write(d.join("gate.jsonl"), "").unwrap();
        std::fs::write(d.join("local.jsonl"), "").unwrap();
        assert_eq!(resolve(&d), d.join("gate.jsonl"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A directory entry named `gate.jsonl` that is not a file must not be
    /// selected — `is_file()` rather than `exists()` is load-bearing.
    #[test]
    fn a_directory_named_like_the_dataset_is_not_the_dataset() {
        let d = scratch("dir");
        std::fs::create_dir_all(d.join("gate.jsonl")).unwrap();
        assert_eq!(resolve(&d), d.join("local.jsonl"));
        let _ = std::fs::remove_dir_all(&d);
    }
}
