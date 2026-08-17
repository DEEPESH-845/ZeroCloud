//! `zc check` — what can this machine run, and how fast?
//!
//! This file no longer renders anything. It assembles the one `Report` that
//! every surface renders, so `--json` and the terminal output can never
//! disagree about a number.

use crate::machine::Machine;
use zc_model::{catalog, predict, Fit, KvPrecision, Verdict};
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
fn rank(row: &Row) -> (u8, i64, i64) {
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
fn best_per_model<'a>(rows: Vec<Row<'a>>) -> Vec<Row<'a>> {
    let mut best: Vec<Row<'a>> = Vec::new();
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
        models = best_per_model(models);
    }
    models.sort_by_key(rank);
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
