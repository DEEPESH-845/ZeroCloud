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
) -> i32 {
    // The catalog is borrowed from by every Row, so it has to outlive them.
    let specs = catalog::load();
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
    let total_rows = models.len();
    if let Some(n) = top {
        models.truncate(n);
    }

    let report = report(m, fit, kv, models, total_rows);
    print!(
        "{}",
        if as_json {
            zc_report::json::render(&report)
        } else {
            zc_report::text::render(&report)
        }
    );
    if as_json {
        println!();
    }
    0
}
