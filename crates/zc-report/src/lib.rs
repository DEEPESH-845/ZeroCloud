//! One result, rendered many ways.
//!
//! Every surface — the terminal, `--json`, and later the HTTP server and the
//! TUI — builds the same [`Report`] and hands it to a renderer here. That is
//! the point: the moment a second surface computes its own view of a
//! prediction, the two drift, and a user comparing `zc check` against
//! `zc serve` finds two different answers with no way to tell which is real.
//!
//! [`Report`] borrows rather than owns. Copying the machine facts into a
//! parallel struct would mean listing every field twice and keeping the copies
//! in sync forever, which is the same drift problem one layer down.

pub mod charset;
pub mod json;
pub mod markdown;
pub mod progress;
pub mod text;

use zc_bench::{compute::ComputeResult, disk::DiskResult, ram::RamResult};
use zc_model::{Backend, Prediction, Quant, Verdict};
use zc_probe::{cpu::Cpu, env::Env, gpu::Gpu, memory::Memory, storage::Storage};

/// One prediction, plus the identity of what was predicted.
pub struct Row<'a> {
    pub model_id: &'a str,
    pub quant: &'a Quant,
    pub prediction: Prediction,
}

/// What the predictions assumed. Stated explicitly because a tok/s figure
/// without its context length and KV precision is not a claim anyone can check.
pub struct Assumptions {
    pub prompt_tokens: u32,
    pub ubatch: u32,
    pub kv_precision: &'static str,
    /// True when the budget used was the idle-machine figure rather than what
    /// is free right now.
    pub idle_machine: bool,
    /// Set when no calibration data backs any coefficient yet.
    pub uncalibrated: bool,
    /// True when no prefill measurement exists for this backend, so TTFT is
    /// reported as unknown rather than derived.
    pub prefill_unmeasured: bool,
    /// How many rows the catalog produced, before `--top` cut it down.
    ///
    /// The cut is applied when the report is *built*, not when it is rendered,
    /// so `--json` and the terminal always describe the same set. A limit that
    /// only existed in one renderer would make the two surfaces disagree about
    /// what a machine can run, which is the drift this crate exists to stop.
    pub total_rows: usize,
}

pub struct Report<'a> {
    pub cpu: &'a Cpu,
    pub mem: &'a Memory,
    pub env: &'a Env,
    pub storage: &'a Storage,
    pub gpus: &'a [Gpu],
    pub ram: &'a RamResult,
    pub compute: &'a ComputeResult,
    /// `None` when the disk measurement failed. Never substituted with a
    /// default: streaming speed is predicted from it, so a stand-in would
    /// produce a confident number with nothing behind it.
    pub disk: Option<&'a DiskResult>,
    pub backend: Backend,
    /// Bandwidth figure actually used for prediction — the performance-core
    /// number on CPU, the all-core peak on unified memory.
    pub ram_bw_gbs: f64,
    /// VRAM bandwidth of the card predictions run against, GB/s. 0 when there
    /// is none. Always a table lookup today, which both renderers must say.
    pub vram_bw_gbs: f64,
    pub vram_bytes: u64,
    pub disk_gbs: f64,
    pub budget_idle: u64,
    pub budget_now: u64,
    pub assumptions: Assumptions,
    pub models: Vec<Row<'a>>,
}

/// Replace the user's home directory with `~`.
///
/// A single substring replacement rather than anything clever, because the
/// failure mode that matters is *missing* an occurrence, not over-matching.
/// `/Users/alice/models/x.gguf` carries a real name into a public bug report.
///
/// Lives at the crate root because four surfaces need it now, not just the
/// Markdown one: `zc fit`, `zc gate` and `zc share` all name the calibration
/// file they read, and `--json` carries the model directory.
pub fn redact(s: &str, home: Option<&str>) -> String {
    match home.filter(|h| !h.is_empty() && *h != "/") {
        Some(h) => s.replace(h, "~"),
        None => s.to_string(),
    }
}

/// The user's home directory, however this platform spells it.
pub fn home() -> Option<String> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
}

/// Convenience for the common case: redact against the real environment.
pub fn redact_home(s: &str) -> String {
    redact(s, home().as_deref())
}

/// Wrap `text` to `width` columns, indenting every line by `indent` spaces.
///
/// Greedy, whitespace-only, and deliberately dumb: a word longer than the
/// width gets its own over-long line rather than being broken, because the
/// long words here are file paths and URLs and a path split across two lines
/// is a path nobody can copy.
pub fn wrap(text: &str, width: usize, indent: usize) -> Vec<String> {
    let pad = " ".repeat(indent);
    let room = width.saturating_sub(indent).max(1);
    let mut out = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if line.is_empty() {
            line.push_str(word);
        } else if line.chars().count() + 1 + word.chars().count() <= room {
            line.push(' ');
            line.push_str(word);
        } else {
            out.push(format!("{pad}{line}"));
            line = word.to_string();
        }
    }
    if !line.is_empty() {
        out.push(format!("{pad}{line}"));
    }
    out
}

/// Stable machine-readable tag for a verdict.
///
/// Separate from `Debug` on purpose: `Debug` output is free to change, and
/// anything a script parses is a promise we have to keep.
pub fn verdict_tag(v: Verdict) -> &'static str {
    match v {
        Verdict::Good => "good",
        Verdict::Usable => "usable",
        Verdict::Slow => "slow",
        Verdict::WontFit => "wont_fit",
    }
}

pub fn backend_tag(b: Backend) -> &'static str {
    match b {
        Backend::Cpu => "cpu",
        Backend::Metal => "metal",
        Backend::Discrete => "discrete",
    }
}

/// Width of the model column, given the widest id actually being shown.
///
/// Floor of 12 keeps the header readable when every row is short. Ceiling of
/// 28 is the widest catalog id today; a longer one overflows its column rather
/// than being truncated, because a truncated model id is not a model id.
///
/// The column used to be a fixed 28. A row that spills to disk carries a
/// `% resident` suffix, which put it at 93 columns -- and a spilling row is
/// exactly what a low-end machine shows, so the table wrapped hardest on the
/// hardware this tool exists for.
pub fn clamp_model_width(widest: usize) -> usize {
    widest.clamp(12, 28)
}

/// Width the model column needs for these rows.
pub fn model_col_width(rows: &[Row]) -> usize {
    clamp_model_width(rows.iter().map(|r| r.model_id.len()).max().unwrap_or(12))
}

/// Sort key: what you can actually run, best first.
///
/// Verdict leads because it is the answer to the question asked — a fast model
/// that will not fit is not a better result than a slow one that will. Then
/// decode speed, then context.
///
/// Deliberately *not* a composite score. llmfit blends speed and fit with a
/// "quality" term built from parameter count and leaderboard opinion; a single
/// number that mixes a measured tok/s with somebody's ranking cannot be
/// defended, and defensibility is the product. Every term here is measured or
/// derived from a measurement.
///
/// Lives here rather than in `zc-cli` because the terminal table and the TUI
/// both order rows with it. The moment two surfaces sort differently, a user
/// comparing them finds two different answers about the same machine.
pub fn rank(row: &Row) -> (u8, i64, i64) {
    let p = &row.prediction;
    let verdict = match p.verdict {
        Verdict::Good => 0,
        Verdict::Usable => 1,
        Verdict::Slow => 2,
        Verdict::WontFit => 3,
    };
    let mid = (p.decode_tok_s.0 + p.decode_tok_s.1) / 2.0;
    (verdict, -(mid * 1000.0) as i64, -(p.max_context as i64))
}

/// Which order the caller wants rows in. The static table only ever uses
/// `Verdict`; the TUI cycles all three.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKey {
    Verdict,
    Decode,
    Context,
}

impl SortKey {
    pub fn next(self) -> SortKey {
        match self {
            SortKey::Verdict => SortKey::Decode,
            SortKey::Decode => SortKey::Context,
            SortKey::Context => SortKey::Verdict,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SortKey::Verdict => "verdict",
            SortKey::Decode => "speed",
            SortKey::Context => "context",
        }
    }
}

/// `rank`, but selectable. `Verdict` returns exactly what `rank` does, so the
/// TUI's default order is byte-identical to the static table's.
///
/// The other two sort by what was asked for, but still sink `WontFit` to the
/// bottom. That is not a hedge: `WontFit` means no context fits alongside the
/// weights, *not* that the weights are too big, so such a row can carry a
/// perfectly good computed decode speed while printing `-`. Sorting purely on
/// that number floats models you cannot run above ones you can — and it only
/// happens on a machine where RAM is tight, which is the machine this tool
/// exists for and not the one it gets developed on.
pub fn rank_by(row: &Row, sort: SortKey) -> (u8, i64, i64) {
    let (v, d, c) = rank(row);
    let unrunnable = u8::from(row.prediction.verdict == Verdict::WontFit);
    match sort {
        SortKey::Verdict => (v, d, c),
        SortKey::Decode => (unrunnable, d, c),
        SortKey::Context => (unrunnable, c, d),
    }
}

/// Keep one row per model: the highest-fidelity quantisation this machine
/// still runs as well as it runs any of them.
///
/// Listing every quantisation of every model buries the answer — the same
/// model appears five times, and the fastest rows are whichever model is
/// smallest. Collapsing needs a rule for "best", and the rule has to stay
/// measured:
///
///   * best verdict first, which is derived from the machine's own numbers;
///   * then the *largest* file, because bytes per parameter is quantisation
///     fidelity and more bits is less quantisation error. That is arithmetic,
///     not a leaderboard opinion.
///
/// So each row is "the best version of this model you can actually run".
pub fn best_per_model(rows: Vec<Row<'_>>) -> Vec<Row<'_>> {
    let mut best: Vec<Row> = Vec::new();
    for row in rows {
        match best.iter().position(|b| b.model_id == row.model_id) {
            None => best.push(row),
            Some(i) => {
                let cur = &best[i];
                let better = rank(&row).0 < rank(cur).0
                    || (rank(&row).0 == rank(cur).0 && row.quant.bytes > cur.quant.bytes);
                if better {
                    best[i] = row;
                }
            }
        }
    }
    best
}

/// `best_per_model`, over indices rather than owned rows.
///
/// The TUI holds every row for the lifetime of the screen and re-derives the
/// visible set on each keystroke, so it sorts and collapses indices rather
/// than moving rows around.
pub fn best_per_model_indices(rows: &[Row], idx: &[usize]) -> Vec<usize> {
    let mut best: Vec<usize> = Vec::new();
    for &i in idx {
        match best.iter().position(|&b| rows[b].model_id == rows[i].model_id) {
            None => best.push(i),
            Some(pos) => {
                let cur = best[pos];
                let better = rank(&rows[i]).0 < rank(&rows[cur]).0
                    || (rank(&rows[i]).0 == rank(&rows[cur]).0
                        && rows[i].quant.bytes > rows[cur].quant.bytes);
                if better {
                    best[pos] = i;
                }
            }
        }
    }
    best
}

/// How a decode range is written. Shared so the terminal table and the TUI can
/// never render the same prediction two different ways.
pub fn fmt_speed(p: &Prediction) -> String {
    if p.verdict == Verdict::WontFit {
        "-".to_string()
    } else {
        format!("{:.1}-{:.1}", p.decode_tok_s.0, p.decode_tok_s.1)
    }
}

pub fn fmt_ctx(p: &Prediction) -> String {
    if p.max_context == 0 {
        "-".to_string()
    } else if p.max_context >= 1024 {
        format!("{}K", p.max_context / 1024)
    } else {
        p.max_context.to_string()
    }
}

/// TTFT is unknown until a real run has been measured on this backend. A dash
/// is honest; a derived number would be wrong by an unknown factor — see
/// `zc-model::fit::prefill_scale`.
pub fn fmt_ttft(p: &Prediction) -> String {
    match p.ttft_s {
        Some(t) if t < 100.0 => format!("{t:.1}s"),
        Some(t) => format!("{t:.0}s"),
        None => "-".to_string(),
    }
}

/// Blank when everything is resident: the eye should land on the rows that
/// spill, and a column of "100%" is noise.
pub fn fmt_resident(p: &Prediction) -> String {
    if p.resident_fraction < 0.999 {
        format!("{:.0}%", p.resident_fraction * 100.0)
    } else {
        String::new()
    }
}

#[cfg(test)]
mod width_tests {
    /// A constrained machine lists short model ids, so the table should narrow
    /// to fit rather than wrapping. The 93-column row that motivated this was
    /// a fixed 28-character column carrying "qwen3-1.7b".
    #[test]
    fn model_column_shrinks_to_the_widest_visible_row() {
        assert_eq!(super::clamp_model_width(10), 12);
        assert_eq!(super::clamp_model_width(20), 20);
        assert_eq!(super::clamp_model_width(40), 28);
    }

    /// The row that motivated this phase was 93 columns: a fixed 28-wide model
    /// column plus a "% resident" suffix. A row that spills is exactly the row
    /// a low-end machine shows, so it is the one that must fit.
    #[test]
    fn a_spilling_row_fits_eighty_columns() {
        let mw = super::clamp_model_width("qwen3-30b-a3b".len());
        let row = format!(
            "  {} {:<mw$} {:<7} {:>12} {:>7} {:>6}  {:<6}{}",
            "OK  ", "qwen3-30b-a3b", "IQ4_XS", "10.7-17.8", "2K", "1.2s", "low", "  89% resident"
        );
        assert!(row.len() <= 80, "row was {} columns: {row}", row.len());
    }

    /// Cycling must return to where it started, or the footer's sort label
    /// drifts out of step with the order on screen.
    #[test]
    fn the_sort_cycle_closes() {
        let s = super::SortKey::Verdict;
        assert_eq!(s.next().next().next(), s);
    }
}

#[cfg(test)]
mod wrap_tests {
    use super::wrap;

    #[test]
    fn wraps_to_the_width_including_the_indent() {
        let t = "the quick brown fox jumps over the lazy dog and keeps running";
        for w in [20usize, 40, 80] {
            for l in wrap(t, w, 2) {
                assert!(l.chars().count() <= w, "{} cols at w={w}: {l}", l.chars().count());
                assert!(l.starts_with("  "));
            }
        }
    }

    /// A path or a URL longer than the width gets its own line rather than
    /// being broken -- a path split across two lines is one nobody can copy.
    #[test]
    fn an_overlong_word_is_never_broken() {
        let long = "~/Library/Application/Support/zerocloud/a/very/long/path.jsonl";
        let out = wrap(&format!("from {long} onwards"), 30, 2);
        assert!(out.iter().any(|l| l.contains(long)), "{out:?}");
        assert_eq!(out.concat().matches("jsonl").count(), 1);
    }

    #[test]
    fn no_line_is_blank_or_padded_only() {
        assert!(wrap("", 80, 2).is_empty());
        for l in wrap("   spaced   out   ", 80, 2) {
            assert_eq!(l.trim_end(), l);
        }
    }
}

#[cfg(test)]
mod sort_tests {
    use super::*;
    use zc_model::spec::QuantFamily;
    use zc_model::{Confidence, Prediction, Quant};

    fn pred(decode: f64, ctx: u32, verdict: Verdict) -> Prediction {
        Prediction {
            resident_fraction: 1.0,
            decode_tok_s: (decode, decode * 1.5),
            prefill_tok_s: None,
            ttft_s: None,
            prefill_confidence: Confidence::Low,
            max_context: ctx,
            kv_bytes_per_token: 1024,
            verdict,
            raw_seconds_per_token: 0.01,
            assumed_eta: 0.8,
            confidence: Confidence::Low,
        }
    }

    /// `WontFit` means no context fits alongside the weights, not that the
    /// weights are too big — so it can carry a high decode speed while the
    /// table prints a dash. Sorting on that number alone put a model the user
    /// cannot run above one they can.
    #[test]
    fn an_unrunnable_model_never_outranks_a_runnable_one() {
        let quant = Quant {
            name: "Q4_K_M".into(),
            bytes: 1 << 30,
            family: QuantFamily::KQuant,
        };
        let fast_but_unrunnable = Row {
            model_id: "no-context-left",
            quant: &quant,
            prediction: pred(200.0, 0, Verdict::WontFit),
        };
        let slow_but_runnable = Row {
            model_id: "actually-works",
            quant: &quant,
            prediction: pred(4.0, 32768, Verdict::Usable),
        };
        for sort in [SortKey::Verdict, SortKey::Decode, SortKey::Context] {
            assert!(
                rank_by(&slow_but_runnable, sort) < rank_by(&fast_but_unrunnable, sort),
                "{sort:?} ranked an unrunnable model first"
            );
        }
    }

    /// Among models that do run, the requested sort still decides.
    #[test]
    fn the_requested_sort_still_orders_runnable_models() {
        let quant = Quant {
            name: "Q4_K_M".into(),
            bytes: 1 << 30,
            family: QuantFamily::KQuant,
        };
        // Worse verdict, but faster and with more context.
        let fast = Row {
            model_id: "fast",
            quant: &quant,
            prediction: pred(50.0, 65536, Verdict::Slow),
        };
        let slow = Row {
            model_id: "slow",
            quant: &quant,
            prediction: pred(5.0, 2048, Verdict::Good),
        };
        assert!(
            rank_by(&slow, SortKey::Verdict) < rank_by(&fast, SortKey::Verdict),
            "verdict sort leads with the verdict"
        );
        assert!(
            rank_by(&fast, SortKey::Decode) < rank_by(&slow, SortKey::Decode),
            "speed sort leads with speed"
        );
        assert!(
            rank_by(&fast, SortKey::Context) < rank_by(&slow, SortKey::Context),
            "context sort leads with context"
        );
    }
}
