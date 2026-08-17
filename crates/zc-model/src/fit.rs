//! Fitting `eta` from calibration records.
//!
//! `zc verify` produces one `implied_eta` per run. This turns a pile of those
//! into the coefficients `predict` uses, and — just as importantly — into an
//! honest *range* around them.
//!
//! Two deliberate statistical choices:
//!
//! **Median, not mean.** A run that happened while the machine was thermally
//! throttled, or while a build was running, is arbitrarily slow. There is no
//! symmetric counterpart: nothing makes a run arbitrarily *fast*. So the error
//! distribution has a long left tail, and a mean would be dragged into it by a
//! single bad sample. The median ignores it.
//!
//! **MAD, not standard deviation.** Same reason: one outlier inflates a
//! standard deviation without bound, which would widen the published range for
//! everyone. Median absolute deviation does not move.
//!
//! The range we publish is a promise. Widening it when we have little data is
//! honest; narrowing it as evidence accumulates is the reward for collecting
//! the dataset.

use crate::json;
use crate::spec::QuantFamily;
use std::collections::HashMap;

/// One measurement, parsed from a line of `data/calibration/*.jsonl`.
#[derive(Debug, Clone)]
pub struct Record {
    pub hw: String,
    pub backend: String,
    pub model: String,
    pub quant: String,
    /// `none` on bare metal; `wsl2`, `container` or `hypervisor` otherwise.
    /// Older records predate the field and read as `None`, which the gate
    /// treats as unknown rather than assuming bare metal.
    pub virt: Option<String>,
    pub quant_family: QuantFamily,
    pub implied_eta: f64,
    /// Signed error of the published range midpoint at the moment the
    /// prediction was made. Absent in older records.
    ///
    /// This is the Phase 0 gate statistic, and it is only meaningful because it
    /// was recorded *before* the run joined the dataset — see [`crate::gate`].
    pub error_pct: Option<f64>,
    /// Whether the measurement landed inside the range we published. The
    /// promise actually made to the user, as opposed to the midpoint error.
    pub within_range: Option<bool>,
    /// Measured prefill throughput expressed against the machine's own
    /// measured f32 FMA peak. Absent in older records.
    ///
    /// Deliberately *not* called an efficiency: it routinely exceeds 1.0,
    /// because real prefill runs on the GPU or through int8 kernels that our
    /// portable f32 benchmark cannot reach. It is a device/path ratio.
    pub implied_prefill_scale: Option<f64>,
}

/// How much evidence stands behind a fitted coefficient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    /// Enough samples that the range is genuinely narrow.
    High,
    Medium,
    Low,
    /// No matching samples at all — the shipped prior, widest range.
    Prior,
}

impl Confidence {
    /// Sample counts at which we allow ourselves to narrow the range.
    ///
    /// Thresholds are conservative on purpose. The failure that kills a
    /// prediction tool is a confident wrong number, so the cost of staying
    /// wide too long is much lower than the cost of narrowing too early.
    fn from_count(n: usize) -> Self {
        match n {
            0 => Confidence::Prior,
            1..=7 => Confidence::Low,
            8..=29 => Confidence::Medium,
            _ => Confidence::High,
        }
    }

    /// Minimum half-width of the published range, as a fraction.
    ///
    /// A floor, not the value: if the observed spread is wider than this, the
    /// observed spread wins.
    ///
    /// The `Prior` figure is the one with evidence behind it, and it is worth
    /// stating where that evidence came from. The first real measurement this
    /// project ever took landed *outside* the range it published: prior eta
    /// 0.616 against a true 0.859, so the truth sat 39.5% above the midpoint of
    /// a range that reached only 30%. A floor that cannot cover the one error
    /// we have measured is not a floor, so it is now 0.40.
    ///
    /// ponytail: fitted to a single observation, which is weak. The upgrade is
    /// to derive it from the dispersion of `implied_eta / assumed_eta` across
    /// machines once there are enough of them to have a distribution — the
    /// shipped prior's error is a measurable quantity, and this constant is a
    /// placeholder for measuring it.
    pub fn min_spread(self) -> f64 {
        match self {
            Confidence::High => 0.08,
            Confidence::Medium => 0.15,
            Confidence::Low => 0.25,
            Confidence::Prior => 0.40,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Confidence::High => "high",
            Confidence::Medium => "medium",
            Confidence::Low => "low",
            Confidence::Prior => "prior",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Coefficient {
    pub eta: f64,
    /// Half-width of the range, as a fraction of `eta`.
    pub spread: f64,
    pub confidence: Confidence,
    pub samples: usize,
}

/// Median of a slice. Takes `&mut` because it sorts in place rather than
/// allocating a copy.
pub(crate) fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// How many standard deviations wide the published range is, each side.
///
/// [`mad`] returns an estimate of sigma, so publishing it directly would give a
/// range covering roughly 68% of runs — a promise that is broken one time in
/// three, which is not a promise. 1.645 is the 95th percentile of a normal
/// distribution, making the two-sided range a 90% interval: one run in ten
/// falls outside, and `within_range` in `zc gate` is the number that proves it.
///
/// Not higher, because a range wide enough never to be wrong tells the user
/// nothing. 90% is the point where the range is still a decision aid.
const COVERAGE: f64 = 1.645;

/// Median absolute deviation, scaled to be comparable with a standard
/// deviation for normally distributed data (the 1.4826 factor).
fn mad(v: &[f64], centre: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut devs: Vec<f64> = v.iter().map(|x| (x - centre).abs()).collect();
    median(&mut devs) * 1.4826
}

pub fn parse_record(line: &str) -> Option<Record> {
    let quant = json::string(line, "quant")?;
    Some(Record {
        hw: json::string(line, "hw")?,
        backend: json::string(line, "backend")?,
        model: json::string(line, "model").unwrap_or_default(),
        virt: json::string(line, "virt"),
        // Records store the GGUF label; the family is what we group by.
        quant_family: QuantFamily::from_gguf_label(&quant),
        quant,
        implied_eta: json::number(line, "implied_eta")?,
        error_pct: json::number(line, "error_pct").filter(|v| v.is_finite()),
        within_range: json::boolean(line, "within_range"),
        implied_prefill_scale: json::number(line, "implied_prefill_scale")
            .filter(|v| v.is_finite() && *v > 0.0),
    })
}

pub fn parse_records(text: &str) -> Vec<Record> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(parse_record)
        .collect()
}

/// Per-backend aggregate, used when a quantisation family has never been
/// measured on that backend.
///
/// Stores the mean *prior* efficiency of the samples that formed it, so a
/// lookup can transfer the ratio rather than the raw value. Pooling raw etas
/// would bias toward whichever family happened to be measured most: a parent
/// built mostly from i-quants would badly overstate a k-quant, and vice versa.
#[derive(Debug, Clone)]
struct Parent {
    eta: f64,
    spread: f64,
    samples: usize,
    mean_prior_efficiency: f64,
}

/// Fitted coefficients, keyed by grouping.
#[derive(Debug, Clone, Default)]
pub struct Fit {
    buckets: HashMap<String, Coefficient>,
    parents: HashMap<String, Parent>,
    /// Prefill scale per backend. Absent means we genuinely do not know, and
    /// callers must report unknown rather than invent a number.
    prefill: HashMap<String, Coefficient>,
}

fn key(backend: &str, family: QuantFamily) -> String {
    format!("{backend}/{}", family.tag())
}

impl Fit {
    /// Group records and fit each bucket.
    ///
    /// Two levels are fitted: the specific `backend/family` bucket, and a
    /// `backend/*` parent that catches quantisations we have not seen on this
    /// backend yet. Falling back to the parent is much better than falling all
    /// the way to a shipped prior.
    pub fn from_records(records: &[Record]) -> Self {
        let mut groups: HashMap<String, Vec<f64>> = HashMap::new();
        let mut parent_vals: HashMap<String, (Vec<f64>, f64)> = HashMap::new();
        for r in records {
            // Guard against garbage: eta above 1.0 means the run beat the
            // machine's measured bandwidth, which is physically impossible and
            // indicates a corrupt record rather than a fast machine.
            if !r.implied_eta.is_finite() || r.implied_eta <= 0.0 || r.implied_eta > 1.5 {
                continue;
            }
            groups
                .entry(key(&r.backend, r.quant_family))
                .or_default()
                .push(r.implied_eta);
            let p = parent_vals.entry(r.backend.clone()).or_default();
            p.0.push(r.implied_eta);
            p.1 += r.quant_family.efficiency();
        }

        let parents = parent_vals
            .into_iter()
            .map(|(backend, (mut vals, eff_sum))| {
                let n = vals.len();
                let eta = median(&mut vals);
                let observed = if eta > 0.0 { COVERAGE * mad(&vals, eta) / eta } else { 0.0 };
                (
                    backend,
                    Parent {
                        eta,
                        spread: observed.max(Confidence::Low.min_spread()),
                        samples: n,
                        mean_prior_efficiency: eff_sum / n.max(1) as f64,
                    },
                )
            })
            .collect();

        // Prefill is grouped by backend only. It is dominated by which device
        // runs the GEMM, not by the weight quantisation.
        let mut prefill_vals: HashMap<String, Vec<f64>> = HashMap::new();
        for r in records {
            if let Some(v) = r.implied_prefill_scale
                && v.is_finite()
                && v > 0.0
                && v < 1000.0
            {
                prefill_vals.entry(r.backend.clone()).or_default().push(v);
            }
        }
        let prefill = prefill_vals
            .into_iter()
            .map(|(backend, mut vals)| {
                let n = vals.len();
                let scale = median(&mut vals);
                let confidence = Confidence::from_count(n);
                let observed = if scale > 0.0 { COVERAGE * mad(&vals, scale) / scale } else { 0.0 };
                (
                    backend,
                    Coefficient {
                        eta: scale,
                        spread: observed.max(confidence.min_spread()),
                        confidence,
                        samples: n,
                    },
                )
            })
            .collect();

        let buckets = groups
            .into_iter()
            .map(|(k, mut vals)| {
                let n = vals.len();
                let eta = median(&mut vals);
                let confidence = Confidence::from_count(n);
                let observed = if eta > 0.0 { COVERAGE * mad(&vals, eta) / eta } else { 0.0 };
                (
                    k,
                    Coefficient {
                        eta,
                        // Never narrower than the confidence floor allows.
                        spread: observed.max(confidence.min_spread()),
                        confidence,
                        samples: n,
                    },
                )
            })
            .collect();

        Fit { buckets, parents, prefill }
    }

    pub fn load(path: &std::path::Path) -> Self {
        std::fs::read_to_string(path)
            .map(|t| Self::from_records(&parse_records(&t)))
            .unwrap_or_default()
    }

    /// Best available coefficient: exact bucket, then backend parent, then the
    /// shipped prior.
    pub fn lookup(&self, backend: &str, family: QuantFamily, prior: f64) -> Coefficient {
        if let Some(c) = self.buckets.get(&key(backend, family)) {
            return c.clone();
        }
        if let Some(p) = self.parents.get(backend) {
            // Transfer the *ratio*, not the value. The parent says this backend
            // achieved `eta` while its samples' priors averaged
            // `mean_prior_efficiency`; apply that same over- or under-shoot to
            // this family's own prior.
            let scale = if p.mean_prior_efficiency > 0.0 {
                family.efficiency() / p.mean_prior_efficiency
            } else {
                1.0
            };
            return Coefficient {
                eta: p.eta * scale,
                // A parent match is real evidence, but not about this
                // quantisation, so it never counts as better than Low.
                confidence: Confidence::Low,
                spread: p.spread.max(Confidence::Low.min_spread()),
                samples: p.samples,
            };
        }
        Coefficient {
            eta: prior,
            spread: Confidence::Prior.min_spread(),
            confidence: Confidence::Prior,
            samples: 0,
        }
    }

    /// Fitted prefill scale for a backend.
    ///
    /// `None` means no run has ever been measured on this backend. There is
    /// deliberately no prior: our compute benchmark cannot see the GPU and
    /// cannot reach the int8 path, so any prior would be a number we know to be
    /// wrong by an unknown factor. Reporting unknown is the honest answer, and
    /// it gives the user a concrete reason to run `zc verify`.
    pub fn prefill_scale(&self, backend: &str) -> Option<Coefficient> {
        self.prefill.get(backend).cloned()
    }

    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    /// Buckets sorted by sample count, for reporting.
    pub fn summary(&self) -> Vec<(String, Coefficient)> {
        let mut v: Vec<_> = self
            .buckets
            .iter()
            .filter(|(k, _)| !k.ends_with("/*"))
            .map(|(k, c)| (k.clone(), c.clone()))
            .collect();
        v.sort_by(|a, b| b.1.samples.cmp(&a.1.samples).then(a.0.cmp(&b.0)));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(backend: &str, quant: &str, eta: f64) -> Record {
        Record {
            hw: "abc".into(),
            backend: backend.into(),
            model: "m".into(),
            quant: quant.into(),
            virt: Some("none".into()),
            quant_family: QuantFamily::from_gguf_label(quant),
            implied_eta: eta,
            error_pct: None,
            within_range: None,
            implied_prefill_scale: None,
        }
    }

    fn rec_p(backend: &str, quant: &str, eta: f64, scale: f64) -> Record {
        Record {
            implied_prefill_scale: Some(scale),
            ..rec(backend, quant, eta)
        }
    }

    /// Prefill must be unknown until measured. A prior would be a number we
    /// already know is wrong by an unknown factor.
    #[test]
    fn prefill_is_unknown_without_measurements() {
        assert!(Fit::default().prefill_scale("Metal").is_none());
        let rs: Vec<Record> = (0..30).map(|_| rec("Metal", "Q4_K_M", 0.85)).collect();
        // Decode records alone must not conjure a prefill scale.
        assert!(Fit::from_records(&rs).prefill_scale("Metal").is_none());
    }

    /// Prefill scale legitimately exceeds 1.0 — the GPU beats our CPU f32
    /// benchmark — so nothing may clamp it into an efficiency-like range.
    #[test]
    fn prefill_scale_may_exceed_one_and_is_grouped_by_backend() {
        let mut rs: Vec<Record> = (0..12).map(|_| rec_p("Metal", "Q4_K_M", 0.85, 8.0)).collect();
        rs.extend((0..3).map(|_| rec_p("Cpu", "Q4_K_M", 0.60, 0.9)));
        let fit = Fit::from_records(&rs);

        let metal = fit.prefill_scale("Metal").expect("metal");
        assert!((metal.eta - 8.0).abs() < 1e-9, "{}", metal.eta);
        assert_eq!(metal.confidence, Confidence::Medium);

        let cpu = fit.prefill_scale("Cpu").expect("cpu");
        assert!((cpu.eta - 0.9).abs() < 1e-9);
        assert_eq!(cpu.confidence, Confidence::Low);

        // No cross-backend fallback: a CPU measurement says nothing about GPU.
        assert!(fit.prefill_scale("Cuda").is_none());
    }

    /// Same robustness requirement as decode: one throttled run must not move
    /// the prefill fit either.
    #[test]
    fn prefill_fit_is_robust_to_outliers() {
        let mut rs: Vec<Record> = (0..9).map(|_| rec_p("Metal", "Q4_K_M", 0.85, 8.0)).collect();
        rs.push(rec_p("Metal", "Q4_K_M", 0.85, 0.4));
        let s = Fit::from_records(&rs).prefill_scale("Metal").expect("metal");
        assert!((s.eta - 8.0).abs() < 1e-9, "{}", s.eta);
    }

    /// The published range is a 90% interval, not a one-sigma band.
    ///
    /// Hand-computed: forty samples split evenly between 0.75 and 0.85 have a
    /// median of 0.80 and a median absolute deviation of exactly 0.05, so
    /// sigma-hat is 0.05 * 1.4826 = 0.074130 and the relative spread must be
    /// 1.645 * 0.074130 / 0.80 = 0.152430. Publishing sigma alone would have
    /// given 0.0927 — a range broken one run in three while claiming to be a
    /// promise.
    #[test]
    fn a_published_range_is_a_ninety_percent_interval_not_one_sigma() {
        let rs: Vec<Record> = (0..40)
            .map(|i| rec("Metal", "Q4_K_M", if i % 2 == 0 { 0.75 } else { 0.85 }))
            .collect();
        let c = Fit::from_records(&rs).lookup("Metal", QuantFamily::KQuant, 0.62);

        assert_eq!(c.confidence, Confidence::High);
        assert!((c.eta - 0.80).abs() < 1e-9, "eta {}", c.eta);
        assert!(
            (c.spread - 0.152_430).abs() < 1e-5,
            "spread {} should be 1.645 sigma, not 1 sigma",
            c.spread
        );
    }

    /// The floor for an uncalibrated machine has to cover the error we have
    /// actually observed in the shipped prior, or the range is decoration.
    ///
    /// From the first real measurement ever taken: assumed eta 0.616 against a
    /// true 0.859, so the truth sat 0.859 / 0.616 - 1 = 39.45% above the
    /// midpoint. The old 30% floor missed it; `within_range` was 0%.
    #[test]
    fn the_prior_range_covers_the_one_miss_on_record() {
        let observed_error = 0.859 / 0.616 - 1.0;
        assert!(
            Confidence::Prior.min_spread() >= observed_error,
            "prior floor {} cannot cover a measured error of {observed_error:.4}",
            Confidence::Prior.min_spread()
        );
        // And an uncalibrated lookup really does publish that floor.
        let c = Fit::default().lookup("Metal", QuantFamily::KQuant, 0.616);
        assert_eq!(c.confidence, Confidence::Prior);
        assert!(c.spread >= observed_error, "spread {}", c.spread);
    }

    #[test]
    fn median_handles_odd_and_even() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&mut []), 0.0);
    }

    /// The whole reason for choosing median over mean: one throttled run must
    /// not move the fit. Here nine samples sit at 0.80 and one at 0.10 —
    /// the mean lands at 0.73, the median stays at 0.80.
    #[test]
    fn median_is_robust_to_a_throttled_outlier() {
        let mut rs: Vec<Record> = (0..9).map(|_| rec("Metal", "Q4_K_M", 0.80)).collect();
        rs.push(rec("Metal", "Q4_K_M", 0.10));

        let fit = Fit::from_records(&rs);
        let c = fit.lookup("Metal", QuantFamily::KQuant, 0.62);
        assert!((c.eta - 0.80).abs() < 1e-9, "eta {}", c.eta);

        let mean: f64 = rs.iter().map(|r| r.implied_eta).sum::<f64>() / rs.len() as f64;
        assert!(mean < 0.74, "mean {mean} should have been dragged down");
    }

    /// Confidence must rise with evidence, and the published range must narrow
    /// as it does.
    #[test]
    fn confidence_and_spread_track_sample_count() {
        for (n, want) in [
            (1, Confidence::Low),
            (7, Confidence::Low),
            (8, Confidence::Medium),
            (29, Confidence::Medium),
            (30, Confidence::High),
        ] {
            let rs: Vec<Record> = (0..n).map(|_| rec("Cpu", "Q4_K_M", 0.7)).collect();
            let c = Fit::from_records(&rs).lookup("Cpu", QuantFamily::KQuant, 0.62);
            assert_eq!(c.confidence, want, "n={n}");
            assert_eq!(c.samples, n);
        }
        let wide = Fit::default().lookup("Cpu", QuantFamily::KQuant, 0.62);
        let narrow: Vec<Record> = (0..40).map(|_| rec("Cpu", "Q4_K_M", 0.7)).collect();
        let narrow = Fit::from_records(&narrow).lookup("Cpu", QuantFamily::KQuant, 0.62);
        assert!(narrow.spread < wide.spread, "{} vs {}", narrow.spread, wide.spread);
    }

    /// Observed spread must win when it is wider than the confidence floor —
    /// plenty of samples that disagree with each other is not confidence.
    #[test]
    fn observed_spread_overrides_the_floor_when_wider() {
        let rs: Vec<Record> = (0..40)
            .map(|i| rec("Cpu", "Q4_K_M", if i % 2 == 0 { 0.4 } else { 0.9 }))
            .collect();
        let c = Fit::from_records(&rs).lookup("Cpu", QuantFamily::KQuant, 0.62);
        assert_eq!(c.confidence, Confidence::High);
        assert!(c.spread > Confidence::High.min_spread(), "spread {}", c.spread);
    }

    /// An unseen quantisation on a known backend falls back to the parent
    /// bucket, but must never be reported as better than Low confidence.
    #[test]
    fn falls_back_to_backend_parent_then_prior() {
        let rs: Vec<Record> = (0..30).map(|_| rec("Metal", "Q4_K_M", 0.85)).collect();
        let fit = Fit::from_records(&rs);

        let exact = fit.lookup("Metal", QuantFamily::KQuant, 0.62);
        assert_eq!(exact.confidence, Confidence::High);

        // Never measured an i-quant on Metal: parent applies, downgraded, and
        // scaled down because i-quants dequantise more slowly than k-quants.
        let parent = fit.lookup("Metal", QuantFamily::IQuant, 0.62);
        assert_eq!(parent.confidence, Confidence::Low);
        let expected = 0.85 * QuantFamily::IQuant.efficiency() / QuantFamily::KQuant.efficiency();
        assert!((parent.eta - expected).abs() < 1e-9, "{} vs {expected}", parent.eta);
        assert!(parent.eta < 0.85, "i-quant must scale below the k-quant parent");

        // Never measured this backend at all: shipped prior.
        let prior = fit.lookup("Cuda", QuantFamily::KQuant, 0.62);
        assert_eq!(prior.confidence, Confidence::Prior);
        assert_eq!(prior.eta, 0.62);
    }

    /// eta above 1.0 means the run beat the machine's own measured bandwidth,
    /// which is impossible. Such records are corrupt and must not pollute a fit.
    #[test]
    fn impossible_records_are_discarded() {
        let rs = vec![
            rec("Cpu", "Q4_K_M", 0.7),
            rec("Cpu", "Q4_K_M", 0.7),
            rec("Cpu", "Q4_K_M", 9.9),
            rec("Cpu", "Q4_K_M", 0.0),
            rec("Cpu", "Q4_K_M", f64::NAN),
            rec("Cpu", "Q4_K_M", -1.0),
        ];
        let c = Fit::from_records(&rs).lookup("Cpu", QuantFamily::KQuant, 0.62);
        assert_eq!(c.samples, 2);
        assert!((c.eta - 0.7).abs() < 1e-9);
    }

    /// Round-trip from the exact JSONL shape `zc verify` writes.
    #[test]
    fn parses_real_record_lines() {
        let text = r#"
{"hw":"deadbeef","os":"macos","backend":"Metal","ram_bw_gbs":132.00,"gflops":420.0,"model":"qwen3:4b","quant":"Q4_K_M","ctx":4096,"actual_decode_tok_s":41.200,"assumed_eta":0.6160,"implied_eta":0.8300,"error_pct":-26.0,"within_range":false}
{"hw":"deadbeef","os":"macos","backend":"Metal","ram_bw_gbs":132.00,"gflops":420.0,"model":"qwen3:30b","quant":"IQ3_XXS","ctx":4096,"actual_decode_tok_s":46.000,"assumed_eta":0.5040,"implied_eta":0.5100,"error_pct":1.2,"within_range":true}
"#;
        let rs = parse_records(text);
        assert_eq!(rs.len(), 2);
        assert_eq!(rs[0].quant_family, QuantFamily::KQuant);
        assert_eq!(rs[1].quant_family, QuantFamily::IQuant);
        assert!((rs[0].implied_eta - 0.83).abs() < 1e-9);

        // The gate reads these two straight off the record; losing them to a
        // parser change would make the Phase 0 metric silently uncomputable.
        assert_eq!(rs[0].error_pct, Some(-26.0));
        assert_eq!(rs[0].within_range, Some(false));
        assert_eq!(rs[1].within_range, Some(true));
        assert_eq!(rs[0].model, "qwen3:4b");

        // Distinct families must not be pooled: that is the whole hypothesis
        // this dataset exists to test.
        let fit = Fit::from_records(&rs);
        let k = fit.lookup("Metal", QuantFamily::KQuant, 0.62);
        let i = fit.lookup("Metal", QuantFamily::IQuant, 0.62);
        assert!((k.eta - 0.83).abs() < 1e-9);
        assert!((i.eta - 0.51).abs() < 1e-9);
    }

    /// The bias this scaling exists to remove: a parent built almost entirely
    /// from slow i-quants must not drag down a fast legacy quant.
    #[test]
    fn parent_fallback_is_not_biased_by_its_family_mix() {
        let rs: Vec<Record> = (0..20).map(|_| rec("Metal", "IQ3_XXS", 0.50)).collect();
        let fit = Fit::from_records(&rs);

        // Legacy quants dequantise far more cheaply than i-quants, so the
        // transferred estimate must be higher than the raw parent median.
        let legacy = fit.lookup("Metal", QuantFamily::Legacy, 0.62);
        assert!(legacy.eta > 0.50, "legacy {} should exceed i-quant parent 0.50", legacy.eta);
        let expected = 0.50 * QuantFamily::Legacy.efficiency() / QuantFamily::IQuant.efficiency();
        assert!((legacy.eta - expected).abs() < 1e-9);

        // Looking up the family the parent is made of must return it unchanged.
        let same = fit.lookup("Metal", QuantFamily::IQuant, 0.62);
        assert!((same.eta - 0.50).abs() < 1e-9);
    }

    #[test]
    fn empty_dataset_yields_prior_only() {
        let fit = Fit::from_records(&[]);
        assert!(fit.is_empty());
        assert_eq!(
            fit.lookup("Metal", QuantFamily::KQuant, 0.62).confidence,
            Confidence::Prior
        );
        assert!(fit.summary().is_empty());
    }

    /// Summary is for humans: most-evidenced first, and parent buckets hidden.
    #[test]
    fn summary_ranks_by_evidence_and_hides_parents() {
        let mut rs: Vec<Record> = (0..5).map(|_| rec("Metal", "Q4_K_M", 0.8)).collect();
        rs.extend((0..12).map(|_| rec("Metal", "IQ3_XXS", 0.5)));
        let s = Fit::from_records(&rs).summary();
        assert_eq!(s.len(), 2, "parent bucket should be hidden");
        assert_eq!(s[0].0, "Metal/i_quant");
        assert_eq!(s[0].1.samples, 12);
    }
}
