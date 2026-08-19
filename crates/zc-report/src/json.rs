//! Machine-readable rendering.
//!
//! Two rules that the text renderer gets to be sloppy about and this one does
//! not:
//!
//! 1. **An unmeasured value is `null`, never a substituted number.** `ttft_s`
//!    and `prefill_tok_s` are `Option` in Rust precisely because deriving them
//!    from a CPU benchmark was measured to be wrong by 10-40x. A consumer that
//!    reads a plausible number where we meant "unknown" has been lied to, and
//!    unlike a human reading a dash it has no way to notice.
//!
//! 2. **A non-finite float is `null`.** JSON has no NaN or Infinity. Emitting
//!    the literal `NaN` produces a document that strict parsers reject, which
//!    turns a bad measurement into a total failure downstream.

use crate::{backend_tag, verdict_tag, Report};
use zc_model::json::escape;
use zc_probe::storage::{Hazard, Medium};

/// Bump when a field changes meaning or disappears. Additive fields do not
/// require a bump; anything a consumer could already be parsing does.
///
/// 2: `storage.model_dir` and `storage.mount` are home-relative (`~/...`)
///    rather than absolute. A user attaching `--json` to a bug report was
///    publishing their account name; a consumer on the same machine can
///    expand `~`, and one on another machine could not have used the absolute
///    path anyway.
const SCHEMA: u32 = 2;

pub fn render(r: &Report) -> String {
    let mut o = String::with_capacity(8192);
    o.push_str(&format!("{{\"schema\":{SCHEMA},"));

    // -- machine ------------------------------------------------------------
    o.push_str("\"machine\":{");
    o.push_str(&format!(
        "\"cpu\":{{\"brand\":{},\"physical\":{},\"logical\":{},\"p_cores\":{},\"e_cores\":{},\
         \"smt\":{},\"llc_bytes\":{},\"cache_line\":{},\"recommended_threads\":{},\
         \"features\":{{\"avx2\":{},\"avx512f\":{},\"f16c\":{},\"neon_dotprod\":{},\"neon_i8mm\":{}}}}},",
        st(&r.cpu.brand),
        r.cpu.physical,
        r.cpu.logical,
        r.cpu.p_cores,
        r.cpu.e_cores,
        r.cpu.smt,
        r.cpu.llc_bytes,
        r.cpu.cache_line,
        r.cpu.recommended_threads,
        r.cpu.features.avx2,
        r.cpu.features.avx512f,
        r.cpu.features.f16c,
        r.cpu.features.neon_dotprod,
        r.cpu.features.neon_i8mm,
    ));
    o.push_str(&format!(
        "\"memory\":{{\"total\":{},\"available\":{},\"page_size\":{},\"unified\":{},\
         \"firmware_reserved\":{},\"gpu_wired_limit\":{}}},",
        r.mem.total,
        r.mem.available,
        r.mem.page_size,
        r.mem.unified,
        opt_u(r.mem.firmware_reserved),
        opt_u(r.mem.gpu_wired_limit),
    ));
    o.push_str(&format!(
        "\"env\":{{\"os\":{},\"arch\":{},\"virt\":{},\"memory_ceiling\":{},\"warning\":{}}},",
        st(r.env.os),
        st(r.env.arch),
        st(r.env.virt_tag()),
        opt_u(r.env.memory_ceiling),
        opt_st(r.env.warning.as_deref()),
    ));
    o.push_str(&format!(
        "\"storage\":{{\"model_dir\":{},\"mount\":{},\"fstype\":{},\"medium\":{},\
         \"source\":{},\"total\":{},\"free\":{},\"bench_file_is_weight\":{},\"hazards\":[",
        st(&crate::redact_home(&r.storage.model_dir.display().to_string())),
        st(&crate::redact_home(&r.storage.mount.display().to_string())),
        st(&r.storage.fstype),
        st(medium_tag(&r.storage.medium)),
        st(r.storage.source),
        r.storage.total,
        r.storage.free,
        r.storage.bench_is_weight,
    ));
    for (i, h) in r.storage.hazards.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push_str(&hazard_json(h));
    }
    o.push_str("]},");

    // `vram_bytes` is per card and 0 for integrated parts — see zc-probe::gpu.
    // `bw_gbs` is looked up rather than measured, and says so, because a
    // consumer that cannot tell the two apart will treat both as measured.
    o.push_str("\"gpus\":[");
    for (i, g) in r.gpus.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push_str(&format!(
            "{{\"name\":{},\"vendor\":{},\"vram_bytes\":{},\"integrated\":{},\"count\":{},\
             \"source\":{},\"bw_gbs\":{},\"bw_measured\":false}}",
            st(&g.name),
            st(g.vendor.tag()),
            g.vram_bytes,
            g.integrated,
            g.count,
            st(g.source),
            num(g.bw_gbs),
        ));
    }
    o.push_str("],");

    o.push_str(&format!(
        "\"budget\":{{\"idle\":{},\"now\":{}}}}},",
        r.budget_idle, r.budget_now
    ));

    // -- measured -----------------------------------------------------------
    o.push_str(&format!(
        "\"measured\":{{\"ram\":{{\"peak_gbs\":{},\"peak_threads\":{},\"working_set\":{},\"by_threads\":[",
        num(r.ram.peak_gbs),
        r.ram.peak_threads,
        r.ram.working_set,
    ));
    for (i, (t, g)) in r.ram.by_threads.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push_str(&format!("{{\"threads\":{t},\"gbs\":{}}}", num(*g)));
    }
    o.push_str("]},");
    o.push_str(&format!(
        "\"compute\":{{\"gflops_1t\":{},\"gflops_nt\":{},\"int8_gops_nt\":{},\"int8_ratio\":{},\"threads\":{}}},",
        num(r.compute.gflops_1t),
        num(r.compute.gflops_nt),
        num(r.compute.int8_gops_nt),
        num(r.compute.int8_ratio),
        r.compute.threads,
    ));
    match r.disk {
        Some(d) => o.push_str(&format!(
            "\"disk\":{{\"seq_qd1_gbs\":{},\"rand_128k_qd1_gbs\":{},\"rand_128k_qdn_gbs\":{},\
             \"rand_4k_qdn_iops\":{},\"queue_depth\":{},\"uncached\":{},\"file_bytes\":{},\"created_file\":{}}}}},",
            num(d.seq_qd1_gbs),
            num(d.rand_128k_qd1_gbs),
            num(d.rand_128k_qdn_gbs),
            num(d.rand_4k_qdn_iops),
            d.queue_depth,
            d.uncached,
            d.file_bytes,
            d.created_file,
        )),
        // Null, not a default. A consumer must be able to tell "the disk is
        // slow" from "we never measured the disk".
        None => o.push_str("\"disk\":null},"),
    }

    // -- assumptions --------------------------------------------------------
    let a = &r.assumptions;
    o.push_str(&format!(
        "\"assumptions\":{{\"backend\":{},\"ram_bw_gbs\":{},\"vram_bytes\":{},\"vram_bw_gbs\":{},\
         \"vram_bw_measured\":false,\"disk_gbs\":{},\"kv_precision\":{},\
         \"prompt_tokens\":{},\"ubatch\":{},\"idle_machine\":{},\"uncalibrated\":{},\"prefill_unmeasured\":{},\"total_rows\":{}}},",
        st(backend_tag(r.backend)),
        num(r.ram_bw_gbs),
        r.vram_bytes,
        num(r.vram_bw_gbs),
        num(r.disk_gbs),
        st(a.kv_precision),
        a.prompt_tokens,
        a.ubatch,
        a.idle_machine,
        a.uncalibrated,
        a.prefill_unmeasured,
        a.total_rows,
    ));

    // -- models -------------------------------------------------------------
    o.push_str("\"models\":[");
    for (i, row) in r.models.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        let p = &row.prediction;
        o.push_str(&format!(
            "{{\"id\":{},\"quant\":{},\"quant_family\":{},\"bytes\":{},\"verdict\":{},\
             \"decode_tok_s\":{{\"low\":{},\"high\":{}}},\"max_context\":{},\"kv_bytes_per_token\":{},\
             \"prefill_tok_s\":{},\"ttft_s\":{},\"confidence\":{},\"prefill_confidence\":{},\
             \"resident_fraction\":{},\"raw_seconds_per_token\":{},\"assumed_eta\":{}}}",
            st(row.model_id),
            st(&row.quant.name),
            st(row.quant.family.tag()),
            row.quant.bytes,
            st(verdict_tag(p.verdict)),
            num(p.decode_tok_s.0),
            num(p.decode_tok_s.1),
            p.max_context,
            p.kv_bytes_per_token,
            opt_num(p.prefill_tok_s),
            opt_num(p.ttft_s),
            st(p.confidence.label()),
            st(p.prefill_confidence.label()),
            num(p.resident_fraction),
            num(p.raw_seconds_per_token),
            num(p.assumed_eta),
        ));
    }
    o.push_str("]}");
    o
}

fn st(s: &str) -> String {
    format!("\"{}\"", escape(s))
}

fn opt_st(s: Option<&str>) -> String {
    s.map_or_else(|| "null".to_string(), st)
}

fn opt_u(v: Option<u64>) -> String {
    v.map_or_else(|| "null".to_string(), |v| v.to_string())
}

/// A finite float, or `null`. JSON cannot represent NaN or Infinity, and a
/// document containing them is rejected outright by strict parsers.
fn num(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.6}")
    } else {
        "null".to_string()
    }
}

fn opt_num(v: Option<f64>) -> String {
    v.map_or_else(|| "null".to_string(), num)
}

fn medium_tag(m: &Medium) -> &'static str {
    match m {
        Medium::Nvme => "nvme",
        Medium::Ssd => "ssd",
        Medium::Rotational => "rotational",
        Medium::Network => "network",
        Medium::Unknown => "unknown",
    }
}

fn hazard_json(h: &Hazard) -> String {
    let (kind, detail) = match h {
        Hazard::CloudSync(s) => ("cloud_sync", Some(s.clone())),
        Hazard::NetworkMount(s) => ("network_mount", Some(s.clone())),
        Hazard::Removable => ("removable", None),
        Hazard::Rotational => ("rotational", None),
        Hazard::LowSpace(b) => ("low_space", Some(b.to_string())),
    };
    format!(
        "{{\"kind\":{},\"detail\":{}}}",
        st(kind),
        opt_st(detail.as_deref())
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of `Option<f64>` on prefill: a consumer must see `null`,
    /// not a plausible-looking number. Deriving TTFT from our f32 CPU benchmark
    /// was measured to be wrong by 10-40x, so a substituted value here would be
    /// confidently wrong in a field nobody can eyeball.
    #[test]
    fn an_unmeasured_value_is_null_not_a_number() {
        assert_eq!(opt_num(None), "null");
        assert_eq!(opt_num(Some(0.0)), "0.000000");
    }

    /// JSON has no NaN or Infinity. A single non-finite float would make the
    /// entire document unparseable, turning one bad measurement into a total
    /// failure for every consumer.
    #[test]
    fn non_finite_floats_become_null() {
        assert_eq!(num(f64::NAN), "null");
        assert_eq!(num(f64::INFINITY), "null");
        assert_eq!(num(f64::NEG_INFINITY), "null");
    }

    /// Model names and mount paths reach this from disk and from the network.
    /// An unescaped quote would not merely corrupt one field, it would let the
    /// value close the string and inject sibling keys.
    #[test]
    fn strings_cannot_break_out_of_their_field() {
        assert_eq!(st(r#"evil","verdict":"good"#), r#""evil\",\"verdict\":\"good""#);
        assert_eq!(st("line\nbreak"), r#""line\nbreak""#);
    }
}
