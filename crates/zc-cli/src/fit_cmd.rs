//! `zc fit` — what the calibration dataset currently says.
//!
//! Makes the evidence behind every prediction inspectable. A user who wants to
//! know why we claim 20 tok/s can see exactly how many real runs that rests on.

use zc_model::{fit::Confidence, Fit};

const DEFAULT_DIR: &str = "data/calibration";
const LOCAL: &str = "local.jsonl";
const CURATED: &str = "gate.jsonl";
const COMMUNITY: &str = "community";

/// Where `zc fit`, `zc gate` and `zc check` *read* the dataset from.
///
/// Overridable so tests and validation runs cannot contaminate a real dataset.
pub fn path() -> std::path::PathBuf {
    env_override().unwrap_or_else(|| resolve(std::path::Path::new(DEFAULT_DIR)))
}

/// Where `zc verify` *writes* a new measurement.
///
/// Always the local file, never the curated one. A measurement is this
/// machine's; promoting it into the cross-machine dataset that backs the
/// published accuracy number is a deliberate, reviewable step
/// (`cat local.jsonl >> gate.jsonl`, or eventually `zc share`) and must never
/// happen as a side effect of running `zc verify`.
///
/// Keeping this separate from [`path`] is load-bearing in CI too: the
/// calibrate workflow uploads `local.jsonl` as its artifact, so a `verify` that
/// wrote anywhere else would silently produce an empty dataset.
pub fn record_path() -> std::path::PathBuf {
    env_override().unwrap_or_else(|| record_in(std::path::Path::new(DEFAULT_DIR)))
}

/// The write target within a calibration directory. Unconditional by design —
/// it takes the same `dir` as [`resolve`] specifically so a test can prove the
/// two disagree when a curated dataset is present.
fn record_in(dir: &std::path::Path) -> std::path::PathBuf {
    dir.join(LOCAL)
}

fn env_override() -> Option<std::path::PathBuf> {
    std::env::var("ZC_CALIBRATION").ok().map(std::path::PathBuf::from)
}

/// Which file in a calibration directory is the dataset to read.
///
/// `gate.jsonl` is the curated cross-machine set committed to the repo — it is
/// what the published accuracy figure is computed from, so CI and any reader
/// can recompute that number from a clean checkout. `local.jsonl` is gitignored
/// and holds whatever this machine's `zc verify` produced. Preferring the
/// committed file when it exists is what makes the claim reproducible; falling
/// back keeps a fresh clone with no dataset behaving exactly as before.
fn resolve(dir: &std::path::Path) -> std::path::PathBuf {
    let curated = dir.join(CURATED);
    if curated.is_file() {
        curated
    } else {
        dir.join(LOCAL)
    }
}

/// Every file the dataset is read from: the curated set, then each merged
/// community submission, sorted so the concatenation is byte-stable across
/// machines and two readers cannot compute different numbers from one repo.
pub fn sources() -> Vec<std::path::PathBuf> {
    match env_override() {
        Some(p) => vec![p],
        None => sources_in(std::path::Path::new(DEFAULT_DIR)),
    }
}

fn sources_in(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = vec![resolve(dir)];
    let mut community: Vec<std::path::PathBuf> = std::fs::read_dir(dir.join(COMMUNITY))
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    community.sort();
    out.extend(community);
    out
}

/// Concatenate the sources, one record per line, dropping exact duplicates.
///
/// Trimming and re-terminating every line is load-bearing: a community file
/// arrives from GitHub's web editor with whatever trailing newline the
/// contributor's browser decided on, and a missing one would weld the next
/// record onto the end of it.
///
/// The dedupe is load-bearing too, and for a reason the single-file version
/// never had to face. `Fit::merge` already drops duplicate lines, so a
/// community file holding a record that is also in `gate.jsonl` was invisible
/// to the coefficients but counted *twice* by `zc gate` — one resubmission of
/// a maintainer's own record would have moved the published accuracy number.
/// Every field in a record is a measurement to three decimal places, so two
/// genuinely distinct runs cannot collide here.
pub fn read_text(paths: &[std::path::PathBuf]) -> String {
    let mut seen = std::collections::HashSet::new();
    let mut out = String::new();
    for p in paths {
        let Ok(text) = std::fs::read_to_string(p) else { continue };
        for line in text.lines() {
            let line = line.trim();
            if !line.is_empty() && seen.insert(line.to_string()) {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out
}

/// What `zc fit` and `zc gate` say they read. Naming the community count
/// separately is the point: it is how a contributor sees their record arrive.
fn describe_sources(paths: &[std::path::PathBuf]) -> String {
    match paths.len() {
        0 => "no dataset".to_string(),
        1 => paths[0].display().to_string(),
        n => format!("{}, plus {} community record(s)", paths[0].display(), n - 1),
    }
}

pub fn run() -> i32 {
    print!("{}", summary_text(&sources()));
    0
}

/// The body of `zc fit`, as text.
///
/// Returned rather than printed so `zc doctor` can embed the identical block.
/// A bug report that shows different coefficients from the ones the user sees
/// would send everyone chasing a discrepancy that is not there.
pub fn summary_text(paths: &[std::path::PathBuf]) -> String {
    use std::fmt::Write;
    let fit = Fit::from_text(&read_text(paths));
    let mut o = String::new();

    if fit.is_empty() {
        let _ = writeln!(o, "No calibration data at {}.\n", describe_sources(paths));
        let _ = writeln!(o, "Every tok/s number zc reports is currently a shipped prior.");
        let _ = writeln!(o, "To replace priors with measurements:");
        let _ = writeln!(o, "    ollama pull qwen3:4b && zc verify");
        return o;
    }

    // Naming a file that does not exist is worse than naming none: an
    // installed user has no repo, so `local.jsonl` is a path they would go
    // looking for. The coefficients they see come from the dataset compiled
    // into the binary, plus their own runs once they have any.
    let source = if paths.first().is_some_and(|p| p.is_file()) {
        format!("{}, plus the dataset shipped in this binary", describe_sources(paths))
    } else {
        "the dataset shipped in this binary".to_string()
    };
    let _ = writeln!(o, "== fitted coefficients ==  ({source})\n");
    let _ = writeln!(
        o,
        "  {:<24} {:>7} {:>9} {:>9} {:>9}  confidence",
        "bucket", "runs", "machines", "eta", "spread"
    );
    for (key, c) in fit.summary() {
        let _ = writeln!(
            o,
            "  {:<24} {:>7} {:>9} {:>9.3} {:>8.0}%  {}",
            key,
            c.samples,
            c.machines,
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
        let _ = writeln!(o, "  8 machines per bucket reaches medium confidence, 30 reaches high.");
        let _ = writeln!(
            o,
            "  More runs on a machine already counted sharpen eta, never the tier:"
        );
        let _ = writeln!(o, "  they measure that machine again, not the next one.");
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

    /// Regression: `zc verify` must never write into the curated dataset.
    ///
    /// When `record_path()` shared `path()`'s resolution, committing
    /// `gate.jsonl` silently redirected every new measurement into the
    /// cross-machine set that backs the published accuracy number — erasing the
    /// line between "what this machine measured" and "the vetted dataset", and
    /// leaving CI's `local.jsonl` artifact empty, which is exactly how it was
    /// caught (calibrate run 32015434922: all seven jobs green, zero records).
    ///
    /// Reads and writes must disagree here: given the *same* directory with a
    /// curated dataset in it, the reader picks it and the writer must not.
    #[test]
    fn verify_writes_to_local_even_when_a_curated_dataset_exists() {
        let d = scratch("split");
        std::fs::write(d.join("gate.jsonl"), "").unwrap();
        assert_eq!(super::resolve(&d), d.join("gate.jsonl"), "reader takes curated");
        assert_eq!(
            super::record_in(&d),
            d.join("local.jsonl"),
            "zc verify must append to local.jsonl, never to the curated gate.jsonl"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A merged submission has to be read without anyone editing a path list,
    /// or the contribution loop needs a maintainer in it twice.
    #[test]
    fn community_records_join_the_curated_file() {
        let d = scratch("community");
        std::fs::write(d.join("gate.jsonl"), "").unwrap();
        std::fs::create_dir_all(d.join("community")).unwrap();
        std::fs::write(d.join("community").join("b-2.jsonl"), "").unwrap();
        std::fs::write(d.join("community").join("a-1.jsonl"), "").unwrap();
        // Not a record file, and must not be read as one.
        std::fs::write(d.join("community").join("README.md"), "").unwrap();

        let got = super::sources_in(&d);
        assert_eq!(
            got,
            vec![
                d.join("gate.jsonl"),
                d.join("community").join("a-1.jsonl"),
                d.join("community").join("b-2.jsonl"),
            ],
            "curated first, then community sorted, and only .jsonl"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The common case is a user with no community directory at all. It must
    /// behave exactly as it did before the directory existed.
    #[test]
    fn a_missing_community_directory_is_not_an_error() {
        let d = scratch("nocommunity");
        std::fs::write(d.join("gate.jsonl"), "").unwrap();
        assert_eq!(super::sources_in(&d), vec![d.join("gate.jsonl")]);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A record already in the curated file, resubmitted as a community file,
    /// must not count twice. `Fit::merge` deduped, `Gate` did not, so the
    /// coefficients ignored the copy while the published accuracy number moved.
    #[test]
    fn a_record_present_in_both_tiers_is_counted_once() {
        let d = scratch("dupe");
        let rec = "{\"hw\":\"a\",\"error_pct\":1.0}";
        std::fs::write(d.join("gate.jsonl"), format!("{rec}\n")).unwrap();
        std::fs::create_dir_all(d.join("community")).unwrap();
        std::fs::write(d.join("community").join("x-1.jsonl"), format!("{rec}\n")).unwrap();
        assert_eq!(super::read_text(&super::sources_in(&d)), format!("{rec}\n"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Concatenation has to normalise line endings: the web editor commits
    /// whatever the contributor's browser sent, and a file with no trailing
    /// newline would otherwise weld two records into one unparseable line.
    #[test]
    fn concatenation_separates_records_that_lack_a_trailing_newline() {
        let d = scratch("concat");
        std::fs::write(d.join("gate.jsonl"), "{\"a\":1}").unwrap();
        std::fs::create_dir_all(d.join("community")).unwrap();
        std::fs::write(d.join("community").join("x-1.jsonl"), "{\"b\":2}\n").unwrap();
        let text = super::read_text(&super::sources_in(&d));
        assert_eq!(text, "{\"a\":1}\n{\"b\":2}\n");
        let _ = std::fs::remove_dir_all(&d);
    }
}
