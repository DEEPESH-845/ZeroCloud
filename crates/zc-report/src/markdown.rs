//! `zc doctor` — everything we read, everything we concluded, in one paste.
//!
//! This exists because the maintainer does not own the machines the code is
//! wrong on. `VERIFICATION.md` lists the Linux and Windows probe paths as never
//! executed and marks the Windows `GetLogicalProcessorInformationEx` walk
//! UNVALIDATED; a bug report that says "it reported 2 cores" is unactionable,
//! and one that carries the raw readings alongside the derived struct is a
//! fix.
//!
//! # Redaction
//!
//! The report is written to be pasted into a public issue, so it must not carry
//! anything identifying. Nothing here collects a hostname, username, serial,
//! MAC or IP in the first place — but paths do leak a username, and the model
//! directory is a path. Every absolute path is therefore rewritten against
//! `$HOME` before it is printed. See [`redact`].

use crate::{redact, Report};
use zc_probe::human;


/// Environment variables that change what `zc` does. Values are shown, because
/// a wrong value is the bug about half the time.
const RELEVANT_ENV: &[&str] = &[
    "ZC_DATA_DIR",
    "OLLAMA_HOST",
    "LLAMA_SERVER_HOST",
    "LMSTUDIO_HOST",
    "OLLAMA_KV_CACHE_TYPE",
    "OLLAMA_FLASH_ATTENTION",
];

pub fn render(r: &Report, fit_summary: &str) -> String {
    let home = crate::home();
    let home = home.as_deref();
    let mut o = String::with_capacity(8192);
    let p = &mut o;

    line(p, "# zc doctor\n");
    line(
        p,
        &format!(
            "`zc {}` on `{}` / `{}`\n",
            env!("CARGO_PKG_VERSION"),
            r.env.os,
            r.env.arch
        ),
    );
    line(
        p,
        "Paste this into a bug report. It contains no hostname, username, \
         serial, MAC or IP; paths are rewritten against `$HOME`.\n",
    );

    // -- what we concluded ---------------------------------------------------
    line(p, "## Concluded\n");
    line(p, "| field | value |");
    line(p, "|---|---|");
    row(p, "cpu", &r.cpu.brand);
    row(
        p,
        "cores",
        &format!(
            "{} physical / {} logical / {}P + {}E / smt {}",
            r.cpu.physical, r.cpu.logical, r.cpu.p_cores, r.cpu.e_cores, r.cpu.smt
        ),
    );
    row(p, "inference threads", &r.cpu.recommended_threads.to_string());
    row(p, "llc", &human(r.cpu.llc_bytes));
    row(
        p,
        "memory",
        &format!(
            "{} total / {} available / unified {}",
            human(r.mem.total),
            human(r.mem.available),
            r.mem.unified
        ),
    );
    row(
        p,
        "firmware reserved",
        &r.mem
            .firmware_reserved
            .map_or_else(|| "unknown".into(), human),
    );
    row(p, "virtualisation", r.env.virt_tag());
    row(p, "backend", crate::backend_tag(r.backend));
    row(
        p,
        "budget",
        &format!("{} idle / {} now", human(r.budget_idle), human(r.budget_now)),
    );
    row(
        p,
        "storage",
        &redact(
            &format!(
                "{} on {} ({:?}) via {}",
                r.storage.mount.display(),
                r.storage.fstype,
                r.storage.medium,
                r.storage.source
            ),
            home,
        ),
    );
    line(p, "");

    if r.gpus.is_empty() {
        line(p, "No GPU detected.\n");
    } else {
        line(p, "| gpu | vram | integrated | count | source | bandwidth |");
        line(p, "|---|---|---|---|---|---|");
        for g in r.gpus {
            line(
                p,
                &format!(
                    "| {} | {} | {} | {} | {} | {} |",
                    g.name,
                    human(g.vram_bytes),
                    g.integrated,
                    g.count,
                    g.source,
                    // An integrated GPU has no VRAM of its own, so it has no
                    // VRAM bandwidth either -- it reads system memory at the
                    // rate zc-bench measured. Printing the family-table figure
                    // here put a fabricated number in the one artifact whose
                    // whole job is to be trustworthy in a bug report, and
                    // disagreed with `zc check`, which has always said
                    // "shares system memory" instead.
                    if g.integrated {
                        "- (shares system memory)".to_string()
                    } else {
                        format!("{:.0} GB/s (looked up)", g.bw_gbs)
                    }
                ),
            );
        }
        line(p, "");
    }

    // -- what we read --------------------------------------------------------
    line(p, "## Raw readings\n");
    line(
        p,
        "The values the table above was derived from. A disagreement between \
         these two sections is the bug.\n",
    );
    line(p, "```");
    for (k, v) in &r.cpu.raw {
        line(p, &format!("{k} = {v}"));
    }
    line(p, &format!("memory.page_size = {}", r.mem.page_size));
    line(
        p,
        &format!(
            "memory.gpu_wired_limit = {}",
            r.mem
                .gpu_wired_limit
                .map_or_else(|| "absent".into(), |v| v.to_string())
        ),
    );
    line(
        p,
        &format!(
            "env.memory_ceiling = {}",
            r.env
                .memory_ceiling
                .map_or_else(|| "absent".into(), |v| v.to_string())
        ),
    );
    line(p, "```\n");

    // -- what we measured ----------------------------------------------------
    line(p, "## Measured\n");
    line(p, "```");
    line(
        p,
        &format!(
            "ram    peak {:.1} GB/s @{}t   working set {}",
            r.ram.peak_gbs,
            r.ram.peak_threads,
            human(r.ram.working_set as u64)
        ),
    );
    // The whole curve, not the peak: a flat curve means the memory controller
    // saturates at one thread, which is what single-channel RAM looks like.
    for (t, g) in &r.ram.by_threads {
        line(p, &format!("       {t:>3}t  {g:>8.1} GB/s"));
    }
    line(
        p,
        &format!(
            "compute {:.1} GFLOPS f32 1t / {:.1} nt @{}t   int8 {:.1} GOPS ({:.2}x)",
            r.compute.gflops_1t,
            r.compute.gflops_nt,
            r.compute.threads,
            r.compute.int8_gops_nt,
            r.compute.int8_ratio
        ),
    );
    match r.disk {
        Some(d) => {
            line(
                p,
                &format!(
                    "disk   seq {:.2} / rand128k qd1 {:.2} / qd{} {:.2} GB/s   4k {:.0} IOPS",
                    d.seq_qd1_gbs, d.rand_128k_qd1_gbs, d.queue_depth, d.rand_128k_qdn_gbs,
                    d.rand_4k_qdn_iops
                ),
            );
            // A cached disk figure is inflated, sometimes by 10x, and every
            // streaming prediction rests on it.
            line(
                p,
                &format!(
                    "       uncached {}   file {}   created {}",
                    d.uncached,
                    human(d.file_bytes),
                    d.created_file
                ),
            );
        }
        None => line(p, "disk   MEASUREMENT FAILED"),
    }
    line(p, "```\n");

    // -- calibration ---------------------------------------------------------
    line(p, "## Calibration\n");
    line(p, "```");
    // Redacted like every other path in this file. `fit_cmd::summary_text`
    // names the calibration file it read, which lives under `$HOME`, so this
    // was the one line in a report headed for a public issue that still
    // carried the user's account name.
    line(p, redact(fit_summary, home).trim_end());
    line(p, "```\n");

    // -- environment ---------------------------------------------------------
    line(p, "## Environment\n");
    line(p, "```");
    let mut any = false;
    for k in RELEVANT_ENV {
        if let Ok(v) = std::env::var(k) {
            line(p, &format!("{k} = {}", redact(&v, home)));
            any = true;
        }
    }
    if !any {
        line(p, "(none of the variables zc reads are set)");
    }
    line(p, "```\n");

    // -- anything we already know is wrong -----------------------------------
    if !r.storage.hazards.is_empty() || r.env.warning.is_some() {
        line(p, "## Warnings\n");
        for h in &r.storage.hazards {
            line(p, &format!("- {h:?}"));
        }
        if let Some(w) = &r.env.warning {
            line(p, &format!("- {w}"));
        }
        line(p, "");
    }

    o
}

fn line(out: &mut String, s: &str) {
    out.push_str(s);
    out.push('\n');
}

fn row(out: &mut String, k: &str, v: &str) {
    line(out, &format!("| {k} | {v} |"));
}

#[cfg(test)]
mod tests {
    use crate::redact;

    /// A bug report is public. `/Users/alice/...` carries a real name into it,
    /// and the model directory is the one field guaranteed to be a path.
    #[test]
    fn home_relative_paths_are_redacted() {
        assert_eq!(
            redact("/Users/alice/models/llama.gguf", Some("/Users/alice")),
            "~/models/llama.gguf"
        );
        assert_eq!(
            redact(r"C:\Users\alice\models", Some(r"C:\Users\alice")),
            r"~\models"
        );
        // A system path outside home is not identifying and stays readable.
        assert_eq!(
            redact("/System/Volumes/Data", Some("/Users/alice")),
            "/System/Volumes/Data"
        );
    }

    /// `HOME=/` would rewrite every path on the machine to `~`, destroying the
    /// report to protect nothing.
    #[test]
    fn a_degenerate_home_is_ignored() {
        assert_eq!(redact("/etc/fstab", Some("/")), "/etc/fstab");
        assert_eq!(redact("/etc/fstab", Some("")), "/etc/fstab");
        assert_eq!(redact("/etc/fstab", None), "/etc/fstab");
    }
}
