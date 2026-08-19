//! Human-readable rendering. This is the output `zc check` has always had.

use crate::Report;
use std::io::IsTerminal;
use zc_model::{Backend, Verdict};
use zc_probe::human;

const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const RESET: &str = "\x1b[0m";

/// Colour only when a human is looking at a terminal that wants it.
///
/// Three conditions, all of them things users legitimately expect: a pipe or a
/// file gets plain text so `zc check > report.txt` and `zc check | grep` are
/// not full of escape codes, `NO_COLOR` is honoured because it is the standard
/// (no-color.org), and `TERM=dumb` means the terminal has said it cannot.
fn colors_enabled() -> bool {
    std::io::stdout().is_terminal()
        && std::env::var_os("NO_COLOR").is_none()
        && std::env::var("TERM").is_ok_and(|t| t != "dumb")
}

/// Wrap already-padded text in a colour.
///
/// Padding first and painting second is the whole contract. `format!("{:<4}")`
/// counts escape bytes as characters, so painting before padding silently
/// shortens the visible cell and every column to its right walks left by five.
fn paint(padded: &str, code: &str, on: bool) -> String {
    if on {
        format!("{code}{padded}{RESET}")
    } else {
        padded.to_string()
    }
}


/// One line if it fits 80 columns, otherwise the variable-length head on a
/// line of its own with the rest indented under it.
///
/// The head here is a CPU brand or a GPU name, and those are the strings this
/// project cannot bound: "Apple M5" is eight characters and
/// "Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz" is forty-five. CI found the
/// hardware line at 104 columns on a runner while it measured 67 here.
fn pair(head: &str, tail: &str) -> Vec<String> {
    let one = format!("  {head}   {tail}");
    if one.chars().count() <= crate::TABLE_WIDTH {
        vec![one]
    } else {
        vec![format!("  {head}"), format!("      {tail}")]
    }
}

fn push_pair(p: &mut String, head: &str, tail: &str) {
    for l in pair(head, tail) {
        push(p, &l);
    }
}

pub fn render(r: &Report) -> String {
    render_with(r, colors_enabled())
}

pub fn render_with(r: &Report, color: bool) -> String {
    let mut o = String::with_capacity(4096);
    let p = &mut o;

    push(p, "== hardware ==");
    push_pair(
        p,
        &r.cpu.brand,
        &format!(
            "{}P+{}E   {} total / {} available{}",
            r.cpu.p_cores,
            r.cpu.e_cores,
            human(r.mem.total),
            human(r.mem.available),
            if r.mem.unified { "   unified" } else { "" }
        ),
    );
    let mount = format!(
        "{} on {} ({})",
        r.storage.mount.display(),
        r.storage.fstype,
        r.storage.medium
    );
    if r.storage.bench_file.is_some() && !r.storage.bench_is_weight {
        push_pair(
            p,
            &mount,
            "[benchmarked against a non-model file on the same volume]",
        );
    } else {
        push(p, &format!("  {mount}"));
    }

    for g in r.gpus {
        let count = if g.count > 1 {
            format!(" x{}", g.count)
        } else {
            String::new()
        };
        push_pair(
            p,
            &format!("{}{}", g.name, count),
            &if g.integrated {
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
            },
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
            "\n== predictions ==  ({} backend, {bw}, KV at {}, {}-token prompt)",
            backend_label(r.backend),
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
    // Unified memory normally means Metal. When it does not, say so unprompted:
    // a Mac user reading "Cpu backend" would otherwise assume a bug, and the
    // difference between the two is roughly an order of magnitude.
    if r.mem.unified && r.backend == Backend::Cpu {
        let saw_adapter = r.gpus.iter().any(|g| g.vendor == zc_probe::gpu::Vendor::Apple);
        push(
            p,
            if saw_adapter {
                "  a GPU was found but reports no shader cores (a VM's paravirtual\n  \
                 adapter), so this predicts CPU decode rather than Metal"
            } else {
                "  no GPU could be verified, so this predicts CPU decode rather than\n  \
                 Metal - run `zc doctor` to see what was probed"
            },
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
    // Widths follow the data, not a constant. Fixed widths were measured on
    // one Apple Silicon laptop and CI on a CPU-only runner overflowed them: no
    // GPU means a 70B streaming off disk reports a TTFT in the thousands of
    // seconds, which is wider than any field this machine ever asked for.
    let cells: Vec<(String, String, String, String, String, String, String)> = r
        .models
        .iter()
        .map(|row| {
            let pr = &row.prediction;
            (
                row.model_id.to_string(),
                row.quant.name.clone(),
                crate::fmt_speed(pr),
                crate::fmt_ctx(pr),
                crate::fmt_ttft(pr),
                pr.confidence.label().to_string(),
                crate::fmt_resident(pr),
            )
        })
        .collect();
    let refs: Vec<(&str, &str, &str, &str, &str, &str, &str)> = cells
        .iter()
        .map(|c| {
            (
                c.0.as_str(),
                c.1.as_str(),
                c.2.as_str(),
                c.3.as_str(),
                c.4.as_str(),
                c.5.as_str(),
                c.6.as_str(),
            )
        })
        .collect();
    let cols = crate::TableCols::for_rows(&refs);
    push(p, &cols.header());

    for (row, c) in r.models.iter().zip(cells.iter()) {
        // The verdict is the one cell the eye should find without reading, so
        // it is the only thing coloured. Painting more would make the table
        // louder without making it faster to scan.
        let (mark, hue) = match row.prediction.verdict {
            Verdict::Good => ("OK", GREEN),
            Verdict::Usable => ("ok", GREEN),
            Verdict::Slow => ("SLOW", YELLOW),
            Verdict::WontFit => ("XX", RED),
        };
        let mark = paint(&format!("{mark:<4}"), hue, color);
        push(
            p,
            &cols.row(&mark, &c.0, &c.1, &c.2, &c.3, &c.4, &c.5, &c.6),
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
                "\n  showing {} of {} - ranked by verdict, then speed, then context\n  --all for every quantisation, --top N to change the cut",
                r.models.len(),
                a.total_rows
            ),
        );
    }

    // Hazards and the environment warning are prose written for a human, and
    // neither has a bounded length -- a WSL memory-ceiling warning names a
    // path. Wrapped, with the continuation indented under the marker.
    for h in r.storage.hazards.iter().map(|h| h.to_string()).chain(r.env.warning.clone()) {
        push(p, "");
        for (i, l) in crate::wrap(&h, crate::TABLE_WIDTH - 5, 0).iter().enumerate() {
            push(p, &format!("{}{l}", if i == 0 { "  !  " } else { "     " }));
        }
    }

    o
}

/// `{:?}` on `Backend` prints `Cpu`/`Metal`/`Discrete`. The first two read as
/// typos to a user and the third names an implementation detail rather than the
/// thing on their desk.
fn backend_label(b: Backend) -> &'static str {
    match b {
        Backend::Cpu => "CPU",
        Backend::Metal => "Metal",
        Backend::Discrete => "discrete GPU",
    }
}

/// Every line goes through here, so trailing whitespace is stripped in one
/// place rather than at each of the dozen `{:<width}` columns that can end a
/// row. Padding after the last visible character breaks copy-paste out of a
/// terminal and shows up as whitespace noise in any report pasted into an
/// issue.
/// Columns a string occupies on screen, ignoring colour escapes.
///
/// `paint` wraps a cell in `\x1b[..m ... \x1b[0m`, which is nine bytes the
/// terminal never draws. Counting them would wrap a row that fits.
fn visible_len(s: &str) -> usize {
    let mut n = 0;
    let mut in_esc = false;
    for c in s.chars() {
        if in_esc {
            in_esc = c != 'm';
        } else if c == '\x1b' {
            in_esc = true;
        } else {
            n += 1;
        }
    }
    n
}

/// Every line of this renderer's output goes through here, so this is the one
/// place that can guarantee the 80-column contract rather than hope for it.
///
/// Two CI failures on machines that could not be reproduced locally came from
/// a line nobody thought to bound — a Xeon brand name, a time-to-first-token
/// in five digits. Each was fixed at its source, and each was found by a user
/// (here, a runner) rather than a test. This is the backstop: anything still
/// too wide is wrapped at a word boundary with a hanging indent, which is
/// strictly better than letting the terminal hard-wrap it mid-word.
fn push(out: &mut String, line: &str) {
    for (i, l) in line.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let l = l.trim_end();
        if visible_len(l) <= crate::TABLE_WIDTH {
            out.push_str(l);
            continue;
        }
        // Preserve the line's own indent so a wrapped row still lines up.
        let indent = l.len() - l.trim_start().len();
        let hang = " ".repeat(indent + 2);
        for (j, part) in crate::wrap(l.trim_start(), crate::TABLE_WIDTH - indent - 2, 0)
            .iter()
            .enumerate()
        {
            if j > 0 {
                out.push('\n');
                out.push_str(&hang);
            } else {
                out.push_str(&" ".repeat(indent));
            }
            out.push_str(part);
        }
    }
    out.push('\n');
}

#[cfg(test)]
mod tests {
    /// The strings this project cannot bound are the CPU brand and the GPU
    /// name. "Apple M5" is eight characters; the Xeon in a GitHub runner is
    /// forty-five. The hardware line measured 67 columns here and 104 there,
    /// and no local run could have shown it.
    #[test]
    fn a_long_cpu_brand_does_not_overrun_the_line() {
        let tail = "4P+0E   16.00 GiB total / 15.00 GiB available";
        for head in [
            "Apple M5",
            "Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz",
            "AMD EPYC 7763 64-Core Processor with an implausibly long marketing name",
        ] {
            for l in super::pair(head, tail) {
                assert!(
                    l.chars().count() <= crate::TABLE_WIDTH,
                    "{} columns: {l}",
                    l.chars().count()
                );
            }
        }
    }

    /// A short head still gets one line — the split is a fallback, not the
    /// default.
    #[test]
    fn a_short_head_stays_on_one_line() {
        assert_eq!(super::pair("Apple M5", "4P+6E   16.00 GiB total").len(), 1);
        assert_eq!(
            super::pair(
                "Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz",
                "4P+0E   16.00 GiB total / 15.00 GiB available"
            )
            .len(),
            2
        );
    }

    /// The backstop. Anything that slips past a per-line fix still cannot
    /// reach a terminal wider than 80 columns.
    #[test]
    fn push_wraps_a_line_nothing_else_bounded() {
        let mut o = String::new();
        super::push(&mut o, &format!("  {}", "word ".repeat(40)));
        for l in o.lines() {
            assert!(l.chars().count() <= crate::TABLE_WIDTH, "{}: {l}", l.chars().count());
        }
        assert!(o.lines().count() > 1);
    }

    /// Colour escapes are bytes the terminal never draws, so they must not
    /// count towards the width -- otherwise a coloured row that fits gets
    /// wrapped for characters nobody can see.
    #[test]
    fn colour_escapes_do_not_count_towards_the_width() {
        let painted = super::paint(&format!("{:<4}", "OK"), super::GREEN, true);
        assert!(painted.chars().count() > 4, "the escapes are really there");
        assert_eq!(super::visible_len(&painted), 4);
        let row = format!("  {painted} {}", "x".repeat(70));
        assert_eq!(super::visible_len(&row), 2 + 4 + 1 + 70);
        let mut o = String::new();
        super::push(&mut o, &row);
        assert_eq!(o.lines().count(), 1, "77 visible columns must not wrap");
    }

    /// Colour must never change a column's width. `format!("{:<4}")` counts the
    /// five bytes of an escape sequence as characters, so painting *before*
    /// padding shortens the visible cell and walks every column to its right
    /// five places left -- a table that looks corrupted only for the users who
    /// have colour on, which is exactly the users who see it first.
    #[test]
    fn colour_never_changes_a_column_width() {
        fn visible(s: &str) -> String {
            let mut out = String::new();
            let mut chars = s.chars();
            while let Some(c) = chars.next() {
                if c == '\x1b' {
                    for c in chars.by_ref() {
                        if c == 'm' {
                            break;
                        }
                    }
                } else {
                    out.push(c);
                }
            }
            out
        }

        let plain = format!("{:<4}", "OK");
        assert_eq!(plain.len(), 4);
        let painted = super::paint(&plain, super::GREEN, true);
        assert_eq!(visible(&painted), plain, "colour changed the visible text");
        assert!(painted.len() > plain.len(), "nothing was actually painted");

        // And a full row keeps its shape: the header and the row must still
        // agree on where the model column starts.
        let off = super::paint(&format!("{:<4}", "SLOW"), super::YELLOW, false);
        assert_eq!(off, "SLOW", "colour off must be byte-identical to plain");
    }

    /// A row whose last column is an empty optional note must not end in the
    /// padding of the column before it.
    #[test]
    fn no_line_ends_in_whitespace() {
        let mut o = String::new();
        super::push(&mut o, &format!("  {:<6}{}", "low", ""));
        super::push(&mut o, "a\n  b   ");
        assert_eq!(o, "  low\na\n  b\n");
    }
}
