//! `zc check` — what can this machine run, and how fast?
//!
//! This file no longer renders anything. It assembles the one `Report` that
//! every surface renders, so `--json` and the terminal output can never
//! disagree about a number.

use crate::machine::Machine;
use zc_model::{catalog, predict, Fit, KvPrecision};
use zc_report::{Assumptions, Report, Row};

const UBATCH: u32 = 512;
const PROMPT: u32 = 2048;

/// Assemble the report every surface renders.
///
/// `zc check` fills `models`; `zc doctor` passes an empty vec because it is a
/// hardware report. Both go through here so no surface can invent its own view
/// of a machine fact.
pub fn report<'a>(
    m: &'a Machine,
    fit: &Fit,
    kv: KvPrecision,
    models: Vec<Row<'a>>,
    total_rows: usize,
) -> Report<'a> {
    Report {
        cpu: &m.cpu,
        mem: &m.mem,
        env: &m.env,
        storage: &m.storage,
        gpus: &m.gpus,
        ram: &m.ram,
        compute: &m.compute,
        disk: m.disk.as_ref(),
        backend: m.backend,
        ram_bw_gbs: m.hw.ram_bw_gbs,
        vram_bw_gbs: m.hw.vram_bw_gbs,
        vram_bytes: m.hw.vram_bytes,
        disk_gbs: m.hw.disk_gbs,
        budget_idle: m.budget_idle,
        budget_now: m.budget_now,
        assumptions: Assumptions {
            prompt_tokens: PROMPT,
            ubatch: UBATCH,
            kv_precision: kv.tag(),
            idle_machine: true,
            uncalibrated: fit.is_empty(),
            prefill_unmeasured: fit.prefill_scale(&format!("{:?}", m.backend)).is_none(),
            total_rows,
        },
        models,
    }
}

pub fn run(
    m: &Machine,
    fit: &Fit,
    kv: KvPrecision,
    top: Option<usize>,
    all_quants: bool,
    as_json: bool,
    tui: bool,
) -> i32 {
    // The catalog is borrowed from by every Row, so it has to outlive them.
    let specs = catalog::load();
    // Built on demand rather than once, because the TUI and the static table
    // want different sets: the TUI needs every quantisation so its `a` key has
    // something to toggle to, and the table wants one row per model. Rebuilding
    // is arithmetic over a few hundred rows and costs nothing measurable.
    let build = |all_quants: bool| {
        let mut models = Vec::new();
        for spec in &specs {
            for quant in &spec.quants {
                models.push(Row {
                    model_id: &spec.id,
                    quant,
                    prediction: predict::predict_with(spec, quant, &m.hw, kv, PROMPT, UBATCH, fit),
                });
            }
        }
        if !all_quants {
            models = zc_report::best_per_model(models);
        }
        models.sort_by_key(zc_report::rank);
        let total = models.len();
        (models, total)
    };

    if tui {
        // Every quantisation, uncollapsed and uncut. The TUI collapses for
        // display and expands again on `a`; handing it a pre-collapsed set
        // made that key inert while the footer and the help overlay both
        // advertised it.
        let (models, total_rows) = build(true);
        let full = report(m, fit, kv, models, total_rows);
        let cs = zc_report::charset::detect();
        // A terminal failure must not cost the user their answer: fall through
        // to the static report rather than exiting.
        if zc_tui::run::run(&full, cs).is_ok() {
            // The alternate screen is gone by now. Print the report the user
            // would have got without a terminal, so the answer is in the
            // scrollback rather than lost with the screen.
            let (mut models, total_rows) = build(all_quants);
            if let Some(n) = top {
                models.truncate(n);
            }
            let plain = report(m, fit, kv, models, total_rows);
            print!("{}", zc_report::text::render(&plain));
            return 0;
        }
    }

    let (mut models, total_rows) = build(all_quants);
    if let Some(n) = top {
        models.truncate(n);
    }
    let plain = report(m, fit, kv, models, total_rows);
    print!(
        "{}",
        if as_json {
            zc_report::json::render(&plain)
        } else {
            zc_report::text::render(&plain)
        }
    );
    if as_json {
        println!();
    }
    0
}

/// `zc check <hf-repo-id>` — a model the catalog does not have.
///
/// Prints the same measured hardware block as `zc check`, then what the
/// repository's own metadata implies about memory. No decode speed: see
/// `hf.rs` for why that is a refusal rather than an omission.
pub fn run_hf(m: &Machine, kv: KvPrecision, repo: &str) -> i32 {
    let fetched = match crate::hf::fetch_model(repo) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    // The hardware block, so the budget the verdict uses is on screen with it.
    let empty = report(m, &Fit::default(), kv, Vec::new(), 0);
    print!("{}", zc_report::text::render(&empty));
    print!(
        "{}",
        crate::hf::render(&fetched, m.budget_idle, kv, UBATCH)
    );
    0
}
