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
const KV: KvPrecision = KvPrecision::Q8;

pub fn run(m: &Machine, fit: &Fit, as_json: bool) -> i32 {
    // The catalog is borrowed from by every Row, so it has to outlive them.
    let specs = catalog::load();
    let mut models = Vec::new();
    for spec in &specs {
        for quant in &spec.quants {
            models.push(Row {
                model_id: &spec.id,
                quant,
                prediction: predict::predict_with(spec, quant, &m.hw, KV, PROMPT, UBATCH, fit),
            });
        }
    }

    let report = Report {
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
            kv_precision: "q8",
            idle_machine: true,
            uncalibrated: fit.is_empty(),
            prefill_unmeasured: fit.prefill_scale(&format!("{:?}", m.backend)).is_none(),
        },
        models,
    };

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
