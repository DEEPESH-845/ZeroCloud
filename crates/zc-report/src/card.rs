//! A result card: one `zc check`, small enough to live in a URL.
//!
//! The payload goes in the URL *fragment* (after `#`), which a browser never
//! sends to a server. The static page that renders it therefore learns nothing
//! the person sharing did not choose to paste, and the host serving it logs
//! only that someone opened `/card/`. That is the whole privacy design, and the
//! reason there is no upload endpoint.
//!
//! What goes in is chosen for sharing, not for debugging: the CPU, memory,
//! the backend and the bandwidth predictions ran against, and the top rows.
//! No paths, no hostname, no serial -- `zc doctor` is the report for a bug.
//!
//! Format `v1.<base64url(JSON)>`. The JSON is versioned by `v` so the page
//! can refuse a shape it does not know rather than render it wrongly:
//!
//! ```text
//! {"v":1,"zc":"0.2.0","kv":"f16","calibrated":true,
//!  "hw":{"cpu":"Apple M5","cores":"4P+6E","ram":17179869184,"gpu":null,
//!        "backend":"metal","bw_gbs":127.0,"budget":13743895348},
//!  "rows":[["qwen3-4b","Q8_0","good",20.8,34.6,40960,"low"], ...]}
//! ```
//!
//! A row is `[id, quant, verdict, decode_low, decode_high, max_context,
//! confidence]`; arrays rather than objects because a URL is the budget.

use crate::{backend_tag, verdict_tag, Report};
use zc_model::json::escape;

/// Rows on a card. Enough to show the shape of what a machine can run; more
/// makes a link long enough that chat apps start truncating it.
pub const MAX_ROWS: usize = 8;

pub fn payload(r: &Report) -> String {
    let gpu = r
        .gpus
        .iter()
        .find(|g| !g.integrated && g.vram_bytes > 0 && g.usable_for_compute())
        .map_or_else(|| "null".to_string(), |g| format!("\"{}\"", escape(&g.name)));
    let rows: Vec<String> = r
        .models
        .iter()
        .take(MAX_ROWS)
        .map(|row| {
            let p = &row.prediction;
            format!(
                "[\"{}\",\"{}\",\"{}\",{},{},{},\"{}\"]",
                escape(row.model_id),
                escape(&row.quant.name),
                verdict_tag(p.verdict),
                num(p.decode_tok_s.0),
                num(p.decode_tok_s.1),
                p.max_context,
                p.confidence.label(),
            )
        })
        .collect();
    let json = format!(
        "{{\"v\":1,\"zc\":\"{}\",\"kv\":\"{}\",\"calibrated\":{},\
         \"hw\":{{\"cpu\":\"{}\",\"cores\":\"{}P+{}E\",\"ram\":{},\"gpu\":{gpu},\
         \"backend\":\"{}\",\"bw_gbs\":{},\"budget\":{}}},\"rows\":[{}]}}",
        env!("CARGO_PKG_VERSION"),
        r.assumptions.kv_precision,
        !r.assumptions.uncalibrated,
        escape(&r.cpu.brand),
        r.cpu.p_cores,
        r.cpu.e_cores,
        r.mem.total,
        backend_tag(r.backend),
        num(r.ram_bw_gbs),
        r.budget_idle,
        rows.join(","),
    );
    format!("v1.{}", base64url(json.as_bytes()))
}

/// One decimal: a card is read by a person, and the range is the claim.
fn num(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.1}")
    } else {
        "null".into()
    }
}

/// RFC 4648 §5, unpadded: the alphabet that needs no escaping in a URL.
fn base64url(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, &b)| n | (b as u32) << (16 - 8 * i));
        for i in 0..=chunk.len() {
            out.push(A[(n >> (18 - 6 * i) & 63) as usize] as char);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::base64url;

    /// RFC 4648 §10's vectors, in the URL alphabet and without padding. The
    /// page decodes with `atob`, so a wrong bit here is an unreadable card.
    #[test]
    fn base64url_matches_the_rfc_vectors() {
        for (plain, enc) in [
            ("", ""),
            ("f", "Zg"),
            ("fo", "Zm8"),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg"),
            ("fooba", "Zm9vYmE"),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64url(plain.as_bytes()), enc);
        }
        // The two characters that differ from standard base64.
        assert_eq!(base64url(&[0xfb, 0xff]), "-_8");
    }
}
