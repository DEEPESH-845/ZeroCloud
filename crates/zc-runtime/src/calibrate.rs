//! Comparing prediction against measurement, and recording the difference.
//!
//! The prediction model is
//!
//! ```text
//! tok_s = eta / raw_seconds_per_token
//! ```
//!
//! where `raw_seconds_per_token` is derived from measured bandwidth and the
//! model's active byte count, and `eta` is the only term we cannot compute
//! from first principles. Inverting a real run gives it directly:
//!
//! ```text
//! implied_eta = actual_tok_s * raw_seconds_per_token
//! ```
//!
//! Accumulating that across models, quantisations and machines *is* the
//! calibration dataset. It is the one asset in this project that cannot be
//! forked away, because it can only be built by running on real hardware.

use crate::ollama::RunStats;
use std::io::Write;
use std::path::Path;
use zc_model::Prediction;

#[derive(Debug, Clone)]
pub struct Calibration {
    pub model: String,
    pub quant: String,
    pub predicted: (f64, f64),
    pub actual: f64,
    /// Signed error of the range midpoint, percent. Negative = we predicted
    /// slower than reality.
    pub error_pct: f64,
    /// Whether the truth landed inside the range we published. This, not
    /// error_pct, is the promise we actually made to the user.
    pub within_range: bool,
    pub assumed_eta: f64,
    pub implied_eta: f64,
    pub actual_prefill_tok_s: f64,
    pub predicted_prefill_tok_s: Option<f64>,
    /// Measured prefill expressed against the machine's own f32 FMA peak.
    ///
    /// ```text
    /// implied_prefill_scale = actual_prefill_tok_s * 2 * active_params / flops
    /// ```
    ///
    /// Routinely exceeds 1.0: real prefill runs on the GPU, or through int8
    /// kernels our portable f32 benchmark cannot reach. It is a device/path
    /// ratio, not an efficiency, and it is the only honest way we have to
    /// predict time-to-first-token.
    pub implied_prefill_scale: f64,
}

/// `active_params` and `gflops` are needed to invert the prefill model; they
/// are not derivable from the prediction alone.
pub fn compare(
    model: &str,
    quant: &str,
    pred: &Prediction,
    run: &RunStats,
    active_params: u64,
    gflops: f64,
) -> Calibration {
    let mid = (pred.decode_tok_s.0 + pred.decode_tok_s.1) / 2.0;
    let actual = run.decode_tok_s;
    let flops = gflops * 1e9;

    Calibration {
        model: model.to_string(),
        quant: quant.to_string(),
        predicted: pred.decode_tok_s,
        actual,
        error_pct: if actual > 0.0 {
            (mid - actual) / actual * 100.0
        } else {
            0.0
        },
        within_range: actual >= pred.decode_tok_s.0 && actual <= pred.decode_tok_s.1,
        assumed_eta: pred.assumed_eta,
        // The whole point of the exercise.
        implied_eta: actual * pred.raw_seconds_per_token,
        actual_prefill_tok_s: run.prefill_tok_s,
        predicted_prefill_tok_s: pred.prefill_tok_s,
        implied_prefill_scale: if flops > 0.0 {
            run.prefill_tok_s * 2.0 * active_params as f64 / flops
        } else {
            0.0
        },
    }
}

/// One line of the calibration dataset.
///
/// Hand-written JSON: the shape is fixed and small, and a serde dependency
/// would be larger than the rest of this crate. Contains no hostname,
/// username, serial or path — a hardware *profile* only.
#[allow(clippy::too_many_arguments)]
pub fn record_line(
    hw_fingerprint: &str,
    os: &str,
    // `none`, `wsl2`, `container` or `hypervisor`. A fixed vocabulary rather
    // than the raw vendor string: it needs no JSON escaping, and the gate only
    // ever asks whether this was bare metal. CI runners are virtual machines,
    // so without this every cloud-collected record would be indistinguishable
    // from one measured on real hardware.
    virt: &str,
    backend: &str,
    ram_bw: f64,
    disk_bw: f64,
    gflops: f64,
    threads: u32,
    ctx: u32,
    active_params: u64,
    cal: &Calibration,
    run: &RunStats,
) -> String {
    format!(
        r#"{{"hw":"{hw}","os":"{os}","virt":"{virt}","backend":"{backend}","ram_bw_gbs":{ram:.2},"disk_bw_gbs":{disk:.2},"gflops":{gf:.1},"threads":{th},"model":"{model}","quant":"{quant}","ctx":{ctx},"prompt_tokens":{pt},"eval_tokens":{et},"predicted_lo":{lo:.3},"predicted_hi":{hi:.3},"actual_decode_tok_s":{act:.3},"actual_prefill_tok_s":{pre:.2},"assumed_eta":{ae:.4},"implied_eta":{ie:.4},"implied_prefill_scale":{ips:.4},"active_params":{ap},"error_pct":{err:.1},"within_range":{wr}}}"#,
        hw = hw_fingerprint,
        os = os,
        virt = virt,
        backend = backend,
        ram = ram_bw,
        disk = disk_bw,
        gf = gflops,
        th = threads,
        model = cal.model,
        quant = cal.quant,
        ctx = ctx,
        pt = run.prompt_tokens,
        et = run.eval_tokens,
        lo = cal.predicted.0,
        hi = cal.predicted.1,
        act = cal.actual,
        pre = cal.actual_prefill_tok_s,
        ae = cal.assumed_eta,
        ie = cal.implied_eta,
        ips = cal.implied_prefill_scale,
        ap = active_params,
        err = cal.error_pct,
        wr = cal.within_range,
    )
}

/// Append a record locally. Nothing leaves the machine without `zc share`.
pub fn append(path: &Path, line: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{line}")
}

/// Stable, non-identifying hash of a hardware profile.
///
/// FNV-1a over the profile string. We need grouping, not security: records
/// from the same machine must land in the same bucket, and the value must not
/// be reversible into anything identifying. No hostname or serial goes in.
pub fn fingerprint(profile: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in profile.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use zc_model::Verdict;

    fn pred(lo: f64, hi: f64, raw: f64, eta: f64) -> Prediction {
        Prediction {
            resident_fraction: 1.0,
            decode_tok_s: (lo, hi),
            prefill_tok_s: Some(100.0),
            ttft_s: Some(1.0),
            max_context: 8192,
            kv_bytes_per_token: 1024,
            verdict: Verdict::Good,
            raw_seconds_per_token: raw,
            assumed_eta: eta,
            confidence: zc_model::Confidence::Prior,
            prefill_confidence: zc_model::Confidence::Prior,
        }
    }

    fn run(decode: f64) -> RunStats {
        RunStats {
            model: "m".into(),
            prompt_tokens: 100,
            eval_tokens: 128,
            prefill_tok_s: 250.0,
            decode_tok_s: decode,
            load_s: 0.1,
        }
    }

    /// The core inversion. Predicted raw time 0.04 s/token (= 25 tok/s at
    /// eta 1.0); measuring 20 tok/s means the machine achieved eta = 0.8.
    #[test]
    fn implied_eta_inverts_the_model() {
        let c = compare("m", "Q4_K_M", &pred(15.0, 25.0, 0.04, 0.62), &run(20.0), 8_000_000_000, 400.0);
        assert!((c.implied_eta - 0.8).abs() < 1e-9, "{}", c.implied_eta);
    }

    /// A prediction that was too low must report negative error, and a
    /// truth inside the published range must be flagged as a hit.
    #[test]
    fn error_sign_and_range_membership() {
        // Range 15-25, midpoint 20, actual 20 -> hit, zero error.
        let hit = compare("m", "q", &pred(15.0, 25.0, 0.04, 0.62), &run(20.0), 8_000_000_000, 400.0);
        assert!(hit.within_range);
        assert!(hit.error_pct.abs() < 1e-9);

        // Predicted 12-20 (mid 16) but actual 25 -> we undershot.
        let low = compare("m", "q", &pred(12.0, 20.0, 0.05, 0.62), &run(25.0), 8_000_000_000, 400.0);
        assert!(!low.within_range);
        assert!(low.error_pct < 0.0, "{}", low.error_pct);
        assert!((low.error_pct + 36.0).abs() < 0.1, "{}", low.error_pct);
    }

    /// A zero measurement must not produce NaN or infinity in the record.
    #[test]
    fn zero_measurement_is_handled() {
        let c = compare("m", "q", &pred(10.0, 20.0, 0.05, 0.62), &run(0.0), 8_000_000_000, 400.0);
        assert_eq!(c.error_pct, 0.0);
        assert!(c.implied_eta.is_finite());
    }

    /// The prefill inversion: 250 tok/s on an 8B model with 400 GFLOPS means
    /// 250 * 2 * 8e9 / 4e11 = 10.0. Above 1.0 by design — the GPU beats our
    /// CPU f32 benchmark, and nothing may clamp that away.
    #[test]
    fn prefill_scale_inverts_and_may_exceed_one() {
        let c = compare("m", "q", &pred(15.0, 25.0, 0.04, 0.62), &run(20.0), 8_000_000_000, 400.0);
        assert!((c.implied_prefill_scale - 10.0).abs() < 1e-9, "{}", c.implied_prefill_scale);
    }

    #[test]
    fn fingerprint_is_stable_and_non_identifying() {
        let a = fingerprint("Apple M5|10|16GiB|macos");
        assert_eq!(a, fingerprint("Apple M5|10|16GiB|macos"));
        assert_ne!(a, fingerprint("Apple M4|10|16GiB|macos"));
        assert_eq!(a.len(), 16);
    }

    /// The record must be one line and must not contain paths or user names.
    #[test]
    fn record_is_single_line_json() {
        let c = compare("qwen3:4b", "Q4_K_M", &pred(15.0, 25.0, 0.04, 0.62), &run(20.0), 8_000_000_000, 400.0);
        let line = record_line("abc", "macos", "none", "Metal", 132.0, 5.0, 420.0, 4, 4096, 8_000_000_000, &c, &run(20.0));
        assert!(!line.contains('\n'));
        assert!(line.starts_with('{') && line.ends_with('}'));
        assert!(line.contains(r#""implied_eta":0.8000"#), "{line}");
        // Without this the gate cannot tell a CI runner from real hardware.
        assert!(line.contains(r#""virt":"none""#), "{line}");
    }
}
