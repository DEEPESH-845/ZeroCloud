//! The Phase 0 gate: is the prediction model good enough to build a product on?
//!
//! ```text
//! median decode prediction error < 25%, across >= 5 machines
//! ```
//!
//! **The error is read from the record, never recomputed.** Every `error_pct`
//! was produced by a prediction made *before* that run joined the dataset, so
//! it is genuinely out-of-sample — the prequential measurement we actually want.
//! Recomputing error against the finished fit would grade the model on its own
//! training data and would flatter it exactly where a wrong answer is most
//! expensive: the decision about whether this product works at all.
//!
//! **The gate statistic is the median of per-machine medians, not the pooled
//! median.** Forty runs on the developer's laptop must not outvote four runs on
//! the four machines the gate exists to test. Both are reported, because a large
//! gap between them is itself the finding — it means accuracy is machine-
//! specific, which is the failure mode this whole project is built to avoid.

use crate::fit::{median, Record};
use std::collections::HashMap;

pub const MAX_MEDIAN_ERROR_PCT: f64 = 25.0;
pub const MIN_MACHINES: usize = 5;

#[derive(Debug, Clone)]
pub struct MachineResult {
    pub hw: String,
    pub runs: usize,
    pub median_abs_error_pct: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Gate {
    /// Records carrying an `error_pct`. Older records without one are excluded
    /// and counted in `skipped`, rather than silently treated as zero error.
    pub runs: usize,
    pub skipped: usize,
    pub machines: Vec<MachineResult>,
    /// Median |error| over every run pooled together.
    pub pooled_median_pct: f64,
    /// Median of each machine's own median. This is the gate.
    pub median_pct: f64,
    /// Share of runs whose measurement landed inside the published range. The
    /// range is the promise; the midpoint is only a summary of it.
    pub within_range_pct: f64,
    /// Worst offenders, largest |error| first, for chasing down.
    pub worst: Vec<(String, String, f64)>,
    pub passed: bool,
}

impl Gate {
    pub fn from_records(records: &[Record]) -> Self {
        let scored: Vec<(&Record, f64)> = records
            .iter()
            .filter_map(|r| r.error_pct.map(|e| (r, e)))
            .collect();
        let skipped = records.len() - scored.len();

        if scored.is_empty() {
            return Gate { skipped, ..Default::default() };
        }

        let mut by_hw: HashMap<&str, Vec<f64>> = HashMap::new();
        for (r, e) in &scored {
            by_hw.entry(r.hw.as_str()).or_default().push(e.abs());
        }
        let mut machines: Vec<MachineResult> = by_hw
            .into_iter()
            .map(|(hw, mut errs)| MachineResult {
                hw: hw.to_string(),
                runs: errs.len(),
                median_abs_error_pct: median(&mut errs),
            })
            .collect();
        machines.sort_by(|a, b| b.runs.cmp(&a.runs).then(a.hw.cmp(&b.hw)));

        let mut per_machine: Vec<f64> = machines.iter().map(|m| m.median_abs_error_pct).collect();
        let median_pct = median(&mut per_machine);

        let mut pooled: Vec<f64> = scored.iter().map(|(_, e)| e.abs()).collect();
        let pooled_median_pct = median(&mut pooled);

        // Only runs that recorded the field count either way, so an older
        // dataset reports a rate over what it actually knows.
        let ranged: Vec<bool> = scored.iter().filter_map(|(r, _)| r.within_range).collect();
        let within_range_pct = if ranged.is_empty() {
            0.0
        } else {
            ranged.iter().filter(|v| **v).count() as f64 / ranged.len() as f64 * 100.0
        };

        let mut worst: Vec<(String, String, f64)> = scored
            .iter()
            .map(|(r, e)| (r.model.clone(), r.quant.clone(), *e))
            .collect();
        worst.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap_or(std::cmp::Ordering::Equal));
        worst.truncate(5);

        Gate {
            runs: scored.len(),
            skipped,
            passed: machines.len() >= MIN_MACHINES && median_pct < MAX_MEDIAN_ERROR_PCT,
            machines,
            pooled_median_pct,
            median_pct,
            within_range_pct,
            worst,
        }
    }

    /// Why the gate is not passed yet, in the order the work should be done.
    pub fn blockers(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.runs == 0 {
            out.push("no measured runs at all - run `zc verify`".into());
            return out;
        }
        let n = self.machines.len();
        if n < MIN_MACHINES {
            out.push(format!(
                "{n} of {MIN_MACHINES} machines - need {} more distinct machine(s)",
                MIN_MACHINES - n
            ));
        }
        if self.median_pct >= MAX_MEDIAN_ERROR_PCT {
            out.push(format!(
                "median error {:.1}% exceeds the {MAX_MEDIAN_ERROR_PCT:.0}% ceiling",
                self.median_pct
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::QuantFamily;

    fn rec(hw: &str, err: f64, within: bool) -> Record {
        Record {
            hw: hw.into(),
            backend: "Metal".into(),
            model: "qwen3:4b".into(),
            quant: "Q4_K_M".into(),
            quant_family: QuantFamily::KQuant,
            implied_eta: 0.8,
            error_pct: Some(err),
            within_range: Some(within),
            implied_prefill_scale: None,
        }
    }

    fn machines(n: usize, err: f64) -> Vec<Record> {
        (0..n).map(|i| rec(&format!("hw{i}"), err, true)).collect()
    }

    /// Accuracy alone is not the gate. One perfectly predicted machine proves
    /// the model fits one machine, which is what the >= 5 requirement exists to
    /// stop us believing.
    #[test]
    fn one_machine_cannot_pass_however_accurate() {
        let rs: Vec<Record> = (0..50).map(|_| rec("only", 0.5, true)).collect();
        let g = Gate::from_records(&rs);
        assert!(!g.passed);
        assert_eq!(g.machines.len(), 1);
        assert!(g.blockers()[0].contains("of 5 machines"), "{:?}", g.blockers());
    }

    /// The reason the gate uses per-machine medians. One well-tuned machine
    /// with a pile of runs drags the pooled median under the ceiling while four
    /// other machines are all badly wrong. Pooling would call that a pass.
    #[test]
    fn a_single_prolific_machine_cannot_carry_the_gate() {
        let mut rs: Vec<Record> = (0..40).map(|_| rec("dev-laptop", 2.0, true)).collect();
        rs.extend((0..4).map(|i| rec(&format!("other{i}"), 60.0, false)));

        let g = Gate::from_records(&rs);
        assert_eq!(g.machines.len(), 5);
        assert!(g.pooled_median_pct < MAX_MEDIAN_ERROR_PCT, "{}", g.pooled_median_pct);
        assert!(g.median_pct >= MAX_MEDIAN_ERROR_PCT, "{}", g.median_pct);
        assert!(!g.passed, "pooling would have passed this; the gate must not");
    }

    #[test]
    fn passes_on_five_accurate_machines() {
        let g = Gate::from_records(&machines(5, 12.0));
        assert!(g.passed);
        assert!((g.median_pct - 12.0).abs() < 1e-9);
        assert!(g.blockers().is_empty());
    }

    /// Sign must not cancel: predicting +40% on one run and -40% on the next is
    /// two bad predictions, not an average of zero.
    #[test]
    fn errors_are_absolute_before_aggregation() {
        let rs = vec![rec("a", 40.0, false), rec("a", -40.0, false)];
        let g = Gate::from_records(&rs);
        assert!((g.pooled_median_pct - 40.0).abs() < 1e-9, "{}", g.pooled_median_pct);
    }

    /// Range membership is the promise made to the user, and is tracked
    /// separately from midpoint error.
    #[test]
    fn within_range_rate_is_reported() {
        let rs = vec![
            rec("a", 5.0, true),
            rec("a", 5.0, true),
            rec("b", 5.0, false),
            rec("b", 5.0, true),
        ];
        assert!((Gate::from_records(&rs).within_range_pct - 75.0).abs() < 1e-9);
    }

    /// A record from before error_pct was written must not be counted as a
    /// perfect prediction.
    #[test]
    fn records_without_an_error_are_skipped_not_zeroed() {
        let mut rs = machines(5, 30.0);
        rs.push(Record { error_pct: None, ..rec("legacy", 0.0, true) });

        let g = Gate::from_records(&rs);
        assert_eq!(g.runs, 5);
        assert_eq!(g.skipped, 1);
        assert_eq!(g.machines.len(), 5, "the unscored machine must not count toward the 5");
        assert!(!g.passed);
    }

    #[test]
    fn empty_dataset_is_a_clean_fail() {
        let g = Gate::from_records(&[]);
        assert!(!g.passed);
        assert_eq!(g.runs, 0);
        assert!(g.median_pct.is_finite());
        assert!(g.blockers()[0].contains("zc verify"));
    }

    #[test]
    fn worst_offenders_are_ranked_by_magnitude() {
        let mut rs = machines(3, 5.0);
        rs.push(Record { model: "llama-3.3-70b".into(), ..rec("a", -88.0, false) });
        let g = Gate::from_records(&rs);
        assert_eq!(g.worst[0].0, "llama-3.3-70b");
        assert!((g.worst[0].2 + 88.0).abs() < 1e-9, "sign is preserved for reading");
    }
}
