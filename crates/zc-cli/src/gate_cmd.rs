//! `zc gate` — does the prediction model clear the Phase 0 bar?
//!
//! The number this prints decides whether the product is worth building. It is
//! deliberately a separate command from `zc fit`: fitting reports what the data
//! says about the coefficients, gating reports how wrong we were *before* that
//! data existed. Reading the second off the first is the mistake this command
//! exists to make impossible.

use zc_model::gate::{Gate, MAX_MEDIAN_ERROR_PCT, MIN_MACHINES};
use zc_model::{fit::all_records, Fit};

/// Non-zero when the gate is not met, so CI and scripts can depend on it.
const FAILED: i32 = 1;

pub fn run() -> i32 {
    let curated = crate::fit_cmd::path();
    let all = crate::fit_cmd::sources();
    // "No data" means no *records*, not a missing file. With two tiers a repo
    // can hold merged community submissions while `gate.jsonl` is absent, and
    // the old `read_to_string` form would have taken this exit while sitting on
    // perfectly good evidence.
    // Against the dataset compiled into this binary, not just the file on disk:
    // an installed user has no repository, and a gate that ignored the shipped
    // records reported `1 of 5 machines` while `zc fit` one line earlier was
    // reading eleven runs.
    let records = all_records(&crate::fit_cmd::read_text(std::slice::from_ref(&curated)));
    if records.is_empty() {
        println!("No calibration data at {}.\n", curated.display());
        println!("The Phase 0 gate is: median decode error < {MAX_MEDIAN_ERROR_PCT:.0}% across >= {MIN_MACHINES} machines.");
        println!("Nothing has been measured, so it is unmeasurable — not failed, unknown.\n");
        println!("    ollama pull qwen3:4b && zc verify");
        return FAILED;
    }

    let g = Gate::from_records(&records);

    println!(
        "== phase 0 gate ==  ({})\n",
        crate::fit_cmd::describe_dataset(std::slice::from_ref(&curated))
    );
    println!(
        "  {} runs on {} machine(s){}",
        g.runs,
        g.machines.len(),
        if g.skipped > 0 {
            format!(", {} older record(s) without an error field skipped", g.skipped)
        } else {
            String::new()
        }
    );

    if g.runs == 0 {
        for b in g.blockers() {
            println!("\n  BLOCKED  {b}");
        }
        return FAILED;
    }

    println!("\n  {:<20} {:>7} {:>18}  kind", "machine", "runs", "median |error|");
    for m in &g.machines {
        println!(
            "  {:<20} {:>7} {:>17.1}%  {}",
            m.hw,
            m.runs,
            m.median_abs_error_pct,
            if m.virtualized { "vm" } else { "bare metal" }
        );
    }

    println!("\n  median of per-machine medians   {:>8.1}%   <- the gate", g.median_pct);
    println!("  pooled median over all runs     {:>8.1}%", g.pooled_median_pct);
    println!("  measurement landed in range     {:>8.1}%   (the promise we published)", g.within_range_pct);

    // The headline number and the exit code come from the curated tier alone.
    // Community records are real evidence and they move every coefficient, but
    // a claim about our own accuracy has to rest on records whose provenance we
    // can state. Printing both, always, is what stops that reading as
    // cherry-picking.
    if all.len() > 1 {
        let g_all = Gate::from_records(&all_records(&crate::fit_cmd::read_text(&all)));
        println!(
            "  {:<32}{:>8.1}%   over {} machine(s)",
            format!("with {} community record(s)", all.len() - 1),
            g_all.median_pct,
            g_all.machines.len()
        );
    }

    // A wide gap means accuracy is machine-specific, which is precisely the
    // failure a hardware-spec-lookup product has and this one is meant not to.
    if (g.median_pct - g.pooled_median_pct).abs() > 10.0 {
        println!("\n  !  per-machine and pooled medians disagree by more than 10 points.");
        println!("     Accuracy is uneven across machines; one machine is carrying the average.");
    }

    if !g.worst.is_empty() {
        println!("\n  worst predictions (negative = we predicted slower than reality)");
        for (model, quant, err) in &g.worst {
            println!("  {:<24} {:<10} {:>+8.1}%", model, quant, err);
        }
    }

    let fit = Fit::from_text(&crate::fit_cmd::read_text(&all));
    if fit.is_empty() {
        println!("\n  note: these errors were produced by shipped priors, not a fit.");
    }

    if g.passed {
        println!(
            "\n  PASS  median {:.1}% < {MAX_MEDIAN_ERROR_PCT:.0}% across {} machines.",
            g.median_pct,
            g.machines.len()
        );
        return 0;
    }
    for b in g.blockers() {
        println!("\n  BLOCKED  {b}");
    }
    FAILED
}
