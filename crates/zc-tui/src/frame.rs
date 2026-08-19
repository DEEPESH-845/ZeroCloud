//! `State` + rows -> the exact lines to paint.
//!
//! Pure, and returns exactly `h` lines each at most `w` columns. Both are
//! load-bearing: a frame that overruns its width makes the terminal wrap,
//! which pushes every later line one row down and scrolls the footer off
//! screen; a frame shorter than `h` leaves the previous screen showing
//! underneath.
//!
//! Renders a [`View`] rather than a `Report` on purpose. `Report` borrows the
//! probed machine, and giving those structs a `Default` so a test could build
//! one would open a path for a fabricated machine fact to reach a prediction —
//! the failure this project's standing rule exists to prevent. A `View` is
//! display-only strings and can never feed the model.

use crate::state::State;
use zc_report::charset::Charset;
use zc_report::Report;

/// One row, already formatted by the shared formatters in `zc-report` so the
/// terminal table and this cannot render the same prediction two ways.
pub struct RowView {
    pub id: String,
    pub quant: String,
    pub speed: String,
    pub ctx: String,
    pub ttft: String,
    pub conf: String,
    /// 1.0 means every weight is in RAM.
    pub resident: f64,
    pub wont_fit: bool,
    // -- detail pane inputs, all already computed by zc-model --
    pub weight_bytes: u64,
    pub eta: f64,
    pub kv_bytes_per_token: u64,
    pub max_context: u32,
}

pub struct View {
    pub cpu: String,
    pub backend: String,
    pub mem_total: u64,
    pub ram_bw_gbs: f64,
    pub disk_gbs: f64,
    pub gflops: f64,
    pub kv: String,
    pub prompt_tokens: u32,
    pub total_rows: usize,
    pub rows: Vec<RowView>,
}

impl View {
    pub fn from_report(r: &Report) -> View {
        View {
            cpu: r.cpu.brand.clone(),
            backend: zc_report::backend_tag(r.backend).to_string(),
            mem_total: r.mem.total,
            ram_bw_gbs: r.ram_bw_gbs,
            disk_gbs: r.disk_gbs,
            gflops: r.compute.gflops_nt,
            kv: r.assumptions.kv_precision.to_string(),
            prompt_tokens: r.assumptions.prompt_tokens,
            total_rows: r.assumptions.total_rows,
            rows: r
                .models
                .iter()
                .map(|row| {
                    let p = &row.prediction;
                    RowView {
                        id: row.model_id.to_string(),
                        quant: row.quant.name.clone(),
                        speed: zc_report::fmt_speed(p),
                        ctx: zc_report::fmt_ctx(p),
                        ttft: zc_report::fmt_ttft(p),
                        conf: p.confidence.label().to_string(),
                        resident: p.resident_fraction,
                        wont_fit: p.verdict == zc_model::Verdict::WontFit,
                        weight_bytes: row.quant.bytes,
                        eta: p.assumed_eta,
                        kv_bytes_per_token: p.kv_bytes_per_token,
                        max_context: p.max_context,
                    }
                })
                .collect(),
        }
    }
}

/// Cut a line to the width, and trim the padding a format string leaves.
///
/// Trailing whitespace is the existing rule for every surface here: it breaks
/// copy-paste out of a terminal and shows up as whitespace diffs in any report
/// a user pastes into an issue.
fn fit(line: &str, w: usize) -> String {
    if line.chars().count() > w {
        line.chars().take(w).collect::<String>().trim_end().to_string()
    } else {
        line.trim_end().to_string()
    }
}

/// Below this there is no honest way to show a table, so say so instead of
/// painting something unreadable.
const MIN_W: usize = 40;
const MIN_H: usize = 10;

/// Lines above the list: two header lines, a rule, and the column header.
const TOP: usize = 4;
/// Lines below the list: a rule and the hint line.
const BOTTOM: usize = 2;

/// Rows the list can show in a window `h` tall. Public so the event loop can
/// scroll by exactly one screen.
pub fn body_height(h: usize) -> usize {
    h.saturating_sub(TOP + BOTTOM)
}

pub fn frame(
    v: &View,
    idx: &[usize],
    total: usize,
    s: &State,
    cs: Charset,
    w: usize,
    h: usize,
) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(h);
    if w < MIN_W || h < MIN_H {
        out.push(fit("terminal too small", w));
        out.push(fit(&format!("need {MIN_W}x{MIN_H}, have {w}x{h}"), w));
        pad_to(&mut out, h, w);
        return out;
    }

    // -- header: the trust anchor. What machine, and what was measured on it.
    out.push(fit(
        &format!(
            "  zc {}  {} {} {} {} {:.0} GiB",
            env!("CARGO_PKG_VERSION"),
            v.cpu,
            cs.sep(),
            v.backend,
            cs.sep(),
            v.mem_total as f64 / (1u64 << 30) as f64,
        ),
        w,
    ));
    out.push(fit(
        &format!(
            "  {:.0} GB/s ram {} {:.1} GB/s disk {} {:.0} GFLOPS {} KV {}",
            v.ram_bw_gbs,
            cs.sep(),
            v.disk_gbs,
            cs.sep(),
            v.gflops,
            cs.sep(),
            v.kv,
        ),
        w,
    ));
    out.push(fit(&cs.rule().repeat(w), w));

    let (mw, cols) = layout(v, idx, w);
    let mut head = format!("    {:<mw$} {:<7} {:>11}", "MODEL", "QUANT", "decode t/s");
    if cols.ctx {
        head.push_str(&format!(" {:>5}", "ctx"));
    }
    if cols.ttft {
        head.push_str(&format!(" {:>5}", "TTFT"));
    }
    if cols.conf {
        head.push_str(&format!(" {:<6}", "conf"));
    }
    if cols.resid {
        head.push_str(&format!(" {:>4}", "%RAM"));
    }
    out.push(fit(&head, w));

    let detail = if s.detail {
        detail_lines(v, idx, s, cs, w)
    } else {
        Vec::new()
    };
    let help = if s.help { help_lines(cs, w) } else { Vec::new() };
    let list_h = body_height(h)
        .saturating_sub(detail.len())
        .saturating_sub(help.len());

    for (screen_i, &row_i) in idx.iter().skip(s.top).take(list_h).enumerate() {
        let r = &v.rows[row_i];
        let dot = if r.wont_fit {
            cs.wont_fit()
        } else if r.resident < 0.999 {
            cs.partial()
        } else {
            cs.resident()
        };
        let resid = if r.resident < 0.999 {
            format!("{:.0}%", r.resident * 100.0)
        } else {
            String::new()
        };
        let mut line = format!(
            "{} {} {:<mw$} {:<7} {:>11}",
            if s.top + screen_i == s.cursor { ">" } else { " " },
            dot,
            r.id,
            r.quant,
            r.speed,
        );
        if cols.ctx {
            line.push_str(&format!(" {:>5}", r.ctx));
        }
        if cols.ttft {
            line.push_str(&format!(" {:>5}", r.ttft));
        }
        if cols.conf {
            line.push_str(&format!(" {:<6}", r.conf));
        }
        if cols.resid {
            line.push_str(&format!(" {:>4}", resid));
        }
        out.push(fit(&line, w));
    }
    if idx.is_empty() {
        out.push(fit("    nothing matches that filter", w));
    }
    out.extend(detail);
    out.extend(help);

    // -- footer
    while out.len() + BOTTOM < h {
        out.push(String::new());
    }
    out.truncate(h.saturating_sub(BOTTOM));
    out.push(fit(&cs.rule().repeat(w), w));
    out.push(fit(&footer(idx, total, s, cs), w));
    pad_to(&mut out, h, w);
    out
}

fn pad_to(out: &mut Vec<String>, h: usize, _w: usize) {
    while out.len() < h {
        out.push(String::new());
    }
    out.truncate(h);
}

/// Which optional columns survive at this width.
///
/// A narrow window used to keep the full 28-column model name and cut the
/// speed off the right edge -- showing the question and hiding the answer.
/// Columns drop in order of what a user can most afford to lose, and the model
/// name gives up width only after every optional column is gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cols {
    pub ctx: bool,
    pub ttft: bool,
    pub conf: bool,
    pub resid: bool,
}

/// Fixed cost of a row: marker, dot, their spaces, quant and speed. Model name
/// and the optional columns are added on top.
const ROW_FIXED: usize = 4 + 8 + 12;

impl Cols {
    fn width(&self) -> usize {
        usize::from(self.ctx) * 6
            + usize::from(self.ttft) * 6
            + usize::from(self.conf) * 7
            + usize::from(self.resid) * 5
    }
}

/// Model column width and the columns that fit, for a window `w` wide.
///
/// The model column follows the rows on screen, so a filtered or constrained
/// view gets a narrow table rather than 28 columns of padding.
pub fn layout(v: &View, idx: &[usize], w: usize) -> (usize, Cols) {
    let widest = idx
        .iter()
        .map(|&i| v.rows[i].id.chars().count())
        .max()
        .unwrap_or(12);
    let mut mw = zc_report::clamp_model_width(widest);
    let mut c = Cols {
        ctx: true,
        ttft: true,
        conf: true,
        resid: true,
    };
    // Confidence goes first: it reads "low" for nearly every row until the
    // calibration dataset grows. Then TTFT, which is often a dash. Then the
    // spill percentage, then context. Speed and the model name never drop --
    // between them they are the whole answer.
    for drop in 0..4 {
        if ROW_FIXED + mw + c.width() <= w {
            break;
        }
        match drop {
            0 => c.conf = false,
            1 => c.ttft = false,
            2 => c.resid = false,
            _ => c.ctx = false,
        }
    }
    // Everything optional is gone and it still does not fit: give back model
    // width down to a floor, so the speed column survives.
    if ROW_FIXED + mw + c.width() > w {
        mw = mw.min(w.saturating_sub(ROW_FIXED + c.width())).max(8);
    }
    (mw, c)
}

fn footer(idx: &[usize], total: usize, s: &State, cs: Charset) -> String {
    if s.filtering {
        return format!("  /{}_", s.filter);
    }
    if s.help {
        return "  ? or esc to close".to_string();
    }
    format!(
        "  {} of {} {} sort {} {} enter why {} / filter {} a quants {} ? keys {} q quit",
        idx.len(),
        total,
        cs.sep(),
        s.sort.label(),
        cs.sep(),
        cs.sep(),
        cs.sep(),
        cs.sep(),
        cs.sep(),
    )
}

fn help_lines(cs: Charset, w: usize) -> Vec<String> {
    let mut o = vec![String::new(), fit(&cs.rule().repeat(w), w)];
    for l in [
        "  up/down or j/k   move            enter   how the number was derived",
        "  pgup/pgdn        move by a page  /       filter by name, esc clears",
        "  home/end         first / last    s       sort: verdict, speed, ctx",
        "  a                every quant     q       quit, printing the report",
    ] {
        o.push(fit(l, w));
    }
    o
}

/// Where the number came from.
///
/// llmfit prints a score; this prints the derivation, which is the whole
/// reason to have measured the machine. Every value is already computed by
/// `zc-model` — nothing is recalculated here, and anything unmeasured prints a
/// dash exactly as it does in the table.
fn detail_lines(v: &View, idx: &[usize], s: &State, cs: Charset, w: usize) -> Vec<String> {
    let Some(&i) = idx.get(s.cursor) else {
        return Vec::new();
    };
    let r = &v.rows[i];
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let weights = gib(r.weight_bytes);
    let spill = weights * (1.0 - r.resident);
    let kv_mib = r.kv_bytes_per_token as f64 / (1u64 << 20) as f64;

    let mut o = vec![
        String::new(),
        fit(&cs.rule().repeat(w), w),
        fit(
            &format!("  {} {} {}   {} tok/s", r.id, cs.sep(), r.quant, r.speed),
            w,
        ),
        String::new(),
        fit(
            &format!(
                "  weights    {weights:>8.2} GiB    {:.0}% resident in RAM",
                r.resident * 100.0
            ),
            w,
        ),
    ];
    if spill > 0.005 {
        o.push(fit(
            &format!(
                "  spill      {spill:>8.2} GiB    streams at {:.1} GB/s",
                v.disk_gbs
            ),
            w,
        ));
    }
    o.push(fit(
        &format!(
            "  bandwidth  {:>8.0} GB/s   measured on this machine",
            v.ram_bw_gbs
        ),
        w,
    ));
    o.push(fit(
        &format!(
            "  eta        {:>8.3}        fitted, confidence {}",
            r.eta, r.conf
        ),
        w,
    ));
    o.push(fit(
        &format!(
            "  context    {:>8}        KV {} at {kv_mib:.2} MiB/token",
            r.ctx, v.kv
        ),
        w,
    ));
    o.push(fit(
        &format!(
            "  TTFT       {:>8}        for a {}-token prompt",
            r.ttft, v.prompt_tokens
        ),
        w,
    ));
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use zc_report::charset::Charset;

    fn row(id: &str, resident: f64, ttft: Option<&str>) -> RowView {
        RowView {
            id: id.to_string(),
            quant: "Q4_K_M".to_string(),
            speed: "10.3-17.2".to_string(),
            ctx: "37K".to_string(),
            ttft: ttft.unwrap_or("-").to_string(),
            conf: "low".to_string(),
            resident,
            wont_fit: false,
            weight_bytes: 8_600_000_000,
            eta: 0.875,
            kv_bytes_per_token: 524_288,
            max_context: 37_000,
        }
    }

    fn view() -> View {
        View {
            cpu: "Apple M5".into(),
            backend: "metal".into(),
            mem_total: 17_179_869_184,
            ram_bw_gbs: 125.0,
            disk_gbs: 5.0,
            gflops: 367.0,
            kv: "F16".into(),
            prompt_tokens: 2048,
            total_rows: 26,
            rows: vec![
                row("qwen3-1.7b", 1.0, Some("0.9s")),
                row("llama-3.1-8b", 0.84, Some("3.4s")),
                row("deepseek-r1-distill-llama-8b", 0.28, None),
            ],
        }
    }

    fn render(w: usize, h: usize, cs: Charset, f: impl Fn(&mut State)) -> Vec<String> {
        let v = view();
        let idx: Vec<usize> = (0..v.rows.len()).collect();
        let mut s = State::new(idx.len());
        f(&mut s);
        let n = idx.len();
        frame(&v, &idx, n, &s, cs, w, h)
    }

    fn plain(w: usize, h: usize) -> Vec<String> {
        render(w, h, Charset::Unicode, |_| {})
    }

    /// A frame that overruns its width corrupts the whole screen: the terminal
    /// wraps, every later line lands one row low, and the footer scrolls off.
    #[test]
    fn no_line_ever_exceeds_the_width_it_was_given() {
        for w in [40usize, 60, 80, 120, 200] {
            for l in plain(w, 24) {
                assert!(
                    l.chars().count() <= w,
                    "{} cols at w={w}: {l}",
                    l.chars().count()
                );
            }
        }
    }

    #[test]
    fn no_line_carries_trailing_whitespace() {
        for st in [0, 1, 2] {
            let f = render(80, 30, Charset::Unicode, |s| {
                s.detail = st == 1;
                s.help = st == 2;
            });
            for l in f {
                assert_eq!(l.trim_end(), l, "trailing whitespace: {l:?}");
            }
        }
    }

    /// A frame shorter than the window leaves the previous screen showing
    /// underneath; a longer one scrolls the header away.
    #[test]
    fn a_frame_is_exactly_as_tall_as_requested() {
        for h in [10usize, 24, 50] {
            assert_eq!(plain(80, h).len(), h, "h={h}");
        }
        for h in [10usize, 24, 50] {
            let f = render(80, h, Charset::Unicode, |s| s.detail = true);
            assert_eq!(f.len(), h, "with detail, h={h}");
        }
    }

    /// Both charsets must lay out identically, or a table built for one wraps
    /// in the other.
    #[test]
    fn ascii_and_unicode_frames_have_the_same_shape() {
        let u = render(80, 24, Charset::Unicode, |s| s.detail = true);
        let a = render(80, 24, Charset::Ascii, |s| s.detail = true);
        assert_eq!(u.len(), a.len());
        for (x, y) in u.iter().zip(a.iter()) {
            assert_eq!(x.chars().count(), y.chars().count(), "\n{x}\n{y}");
        }
    }

    #[test]
    fn a_tiny_terminal_gets_a_message_not_a_broken_table() {
        let f = plain(20, 6);
        assert_eq!(f.len(), 6);
        assert!(f.iter().any(|l| l.contains("too small")), "{f:?}");
    }

    /// The pane is the product thesis made interactive: it must show where the
    /// number came from, not just the number.
    #[test]
    fn the_detail_pane_shows_the_derivation() {
        let f = render(100, 30, Charset::Unicode, |s| s.detail = true).join("\n");
        for want in ["weights", "bandwidth", "eta", "context", "resident"] {
            assert!(f.contains(want), "detail pane missing {want:?}:\n{f}");
        }
    }

    /// An unmeasured value is a dash, never a substituted constant.
    #[test]
    fn an_unmeasured_field_is_a_dash_in_the_pane() {
        let f = render(100, 30, Charset::Unicode, |s| {
            s.detail = true;
            s.cursor = 2; // the row whose TTFT is None
        });
        // Match the pane's own line, not the table's column header, which
        // also says TTFT and would pass this assertion for the wrong reason.
        let ttft = f
            .iter()
            .find(|l| l.starts_with("  TTFT "))
            .expect("no TTFT line in the pane");
        let value = ttft.split_whitespace().nth(1).unwrap();
        assert_eq!(value, "-", "unmeasured TTFT must print a dash: {ttft}");
    }

    /// A model that spills is the interesting case; the pane must say how much
    /// and how fast it streams.
    #[test]
    fn the_pane_reports_spill_only_when_there_is_some() {
        let spills = render(100, 30, Charset::Unicode, |s| {
            s.detail = true;
            s.cursor = 1;
        })
        .join("\n");
        assert!(spills.contains("spill"), "{spills}");
        let resident = render(100, 30, Charset::Unicode, |s| {
            s.detail = true;
            s.cursor = 0;
        })
        .join("\n");
        assert!(
            !resident.contains("spill"),
            "a fully resident model has no spill line:\n{resident}"
        );
    }

    /// Every key must appear in the overlay, or the overlay is a lie.
    #[test]
    fn the_help_overlay_lists_every_key() {
        let f = render(100, 30, Charset::Unicode, |s| s.help = true).join("\n");
        for k in ["enter", "/", "s", "a", "q", "pgup", "home"] {
            assert!(f.contains(k), "help overlay missing {k:?}:\n{f}");
        }
    }

    /// A filter matching nothing must say so rather than showing a blank body
    /// the user reads as a crash.
    #[test]
    fn an_empty_result_set_says_so() {
        let v = view();
        let mut s = State::new(0);
        s.filter = "zzz".into();
        let f = frame(&v, &[], 3, &s, Charset::Unicode, 80, 24);
        assert!(f.iter().any(|l| l.contains("nothing matches")), "{f:?}");
        assert_eq!(f.len(), 24);
    }

    /// A narrow window used to keep the full model name and cut the speed off
    /// the right edge -- showing the question and hiding the answer. Speed is
    /// the answer, so it survives every width the table renders at.
    #[test]
    fn the_speed_column_survives_every_width() {
        for w in [40usize, 48, 55, 62, 70, 80, 120] {
            let f = plain(w, 24);
            let row = f
                .iter()
                .find(|l| l.contains("qwen3-1.7b"))
                .unwrap_or_else(|| panic!("no model row at w={w}:\n{}", f.join("\n")));
            assert!(
                row.contains("10.3-17.2"),
                "speed cut off at w={w}: {row:?}"
            );
        }
    }

    /// Columns drop in order of what a user can most afford to lose, and the
    /// set only ever shrinks as the window narrows.
    #[test]
    fn columns_drop_in_a_fixed_order_as_the_window_narrows() {
        let v = view();
        let idx: Vec<usize> = (0..v.rows.len()).collect();
        let wide = layout(&v, &idx, 120).1;
        assert_eq!(
            wide,
            Cols { ctx: true, ttft: true, conf: true, resid: true },
            "everything fits at 120"
        );
        let mut prev = wide;
        for w in (40..=120).rev() {
            let c = layout(&v, &idx, w).1;
            for (now, before) in [
                (c.ctx, prev.ctx),
                (c.ttft, prev.ttft),
                (c.conf, prev.conf),
                (c.resid, prev.resid),
            ] {
                assert!(!now || before, "a column came back at w={w}");
            }
            prev = c;
        }
        // Confidence is the first to go, context the last.
        let narrow = layout(&v, &idx, 60).1;
        assert!(!narrow.conf, "conf should drop before ctx");
    }

    /// The cursor marker must be on exactly one row, or the user cannot tell
    /// what enter will open.
    #[test]
    fn exactly_one_row_carries_the_cursor() {
        let f = render(80, 24, Charset::Unicode, |s| s.cursor = 1);
        assert_eq!(f.iter().filter(|l| l.starts_with('>')).count(), 1);
    }
}
