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

pub fn render(r: &Report) -> String {
    render_with(r, colors_enabled())
}

pub fn render_with(r: &Report, color: bool) -> String {
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
            "  {} on {} ({}){}",
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
    // Width follows the rows actually shown. Fixed at 28 this table ran to 93
    // columns on any row that spills to disk -- which is the row a low-end
    // machine shows, so it wrapped hardest on the hardware this tool is for.
    let mw = crate::model_col_width(&r.models);
    push(
        p,
        &format!(
            "  {:<4} {:<mw$} {:<7} {:>11} {:>5} {:>5} {:<6} {:>4}",
            "", "model", "quant", "decode t/s", "ctx", "TTFT", "conf", "%RAM"
        ),
    );

    for row in &r.models {
        let pr = &row.prediction;
        // The verdict is the one cell the eye should find without reading, so
        // it is the only thing coloured. Painting more would make the table
        // louder without making it faster to scan.
        let (mark, hue) = match pr.verdict {
            Verdict::Good => ("OK", GREEN),
            Verdict::Usable => ("ok", GREEN),
            Verdict::Slow => ("SLOW", YELLOW),
            Verdict::WontFit => ("XX", RED),
        };
        let mark = paint(&format!("{mark:<4}"), hue, color);
        // Shared with the TUI so one prediction cannot render two ways.
        let ctx = crate::fmt_ctx(pr);
        let speed = crate::fmt_speed(pr);
        let ttft = crate::fmt_ttft(pr);
        push(
            p,
            &format!(
                "  {} {:<mw$} {:<7} {:>11} {:>5} {:>5} {:<6} {:>4}",
                mark,
                row.model_id,
                row.quant.name,
                speed,
                ctx,
                ttft,
                pr.confidence.label(),
                // A column rather than a sentence. "  89% resident" cost 14
                // columns and pushed a spilling row to 93; "89%" under a %RAM
                // header says the same thing in four and fits at any model
                // width the catalog can produce.
                //
                // Blank rather than "100%" when everything is resident: the
                // eye should land on the rows that spill, and `push` trims the
                // trailing space so a full row still ends where it stops.
                crate::fmt_resident(pr)
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
                "\n  showing {} of {} - ranked by verdict, then speed, then context\n  --all for every quantisation, --top N to change the cut",
                r.models.len(),
                a.total_rows
            ),
        );
    }

    for h in &r.storage.hazards {
        push(p, &format!("\n  !  {h}"));
    }
    if let Some(w) = &r.env.warning {
        push(p, &format!("\n  !  {w}"));
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
fn push(out: &mut String, line: &str) {
    for (i, l) in line.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(l.trim_end());
    }
    out.push('\n');
}

#[cfg(test)]
mod tests {
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
