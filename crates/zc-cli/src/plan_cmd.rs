//! `zc plan` — what would it take to run this?
//!
//! The inverse of `zc check`. `check` starts from the machine and asks what
//! fits; this starts from the model and asks what running it would need.
//!
//! The differentiated line is the bandwidth. llmfit answers "what hardware do
//! I need" with a GPU model name; a name is a lookup, and a lookup is what this
//! project refuses to put underneath a number. Bandwidth is the unit that
//! actually governs decode — it falls out of inverting `decode = eta ·
//! bandwidth / active_bytes`, and anyone can check it against a spec sheet.
//!
//! Because this machine has been measured, the plan can also say how far short
//! it falls, which is something no tool that never measured can express.

use crate::machine::Machine;
use zc_model::{catalog, predict, Fit, KvPrecision};

/// Ollama's default, and the number most people are actually running at.
/// Stated in the header so the plan is never read as context-free.
const DEFAULT_CTX: u32 = 4096;
/// `predict.rs` calls 10 tok/s the boundary of comfortable for interactive
/// chat. Borrowed rather than invented so the default target means something.
const DEFAULT_TPS: f64 = 10.0;
const UBATCH: u32 = 512;

/// Parse `32768`, `32k` or `32K`.
pub fn parse_ctx(s: &str) -> Option<u32> {
    let t = s.trim();
    let (num, mult) = match t.strip_suffix(['k', 'K']) {
        Some(n) => (n, 1024u64),
        None => (t, 1),
    };
    let n: u64 = num.parse().ok()?;
    let v = n.checked_mul(mult)?;
    if v == 0 || v > u32::MAX as u64 {
        return None;
    }
    Some(v as u32)
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// Find one catalog entry by exact id, else by unique prefix.
///
/// Refuses an ambiguous prefix rather than picking one: planning for the wrong
/// model silently is worse than asking again.
fn resolve<'a>(specs: &'a [zc_model::ModelSpec], want: &str) -> Result<&'a zc_model::ModelSpec, String> {
    let want_l = want.to_ascii_lowercase();
    if let Some(s) = specs.iter().find(|s| s.id.eq_ignore_ascii_case(want)) {
        return Ok(s);
    }
    let hits: Vec<&zc_model::ModelSpec> = specs
        .iter()
        .filter(|s| s.id.to_ascii_lowercase().contains(&want_l))
        .collect();
    match hits.len() {
        1 => Ok(hits[0]),
        0 => Err(format!(
            "no catalog model matches '{want}'. `zc check --all` lists every one,\n\
             and `zc check <hf-repo-id>` answers for a model that is not in it."
        )),
        _ => {
            let names: Vec<&str> = hits.iter().map(|s| s.id.as_str()).take(8).collect();
            Err(format!(
                "'{want}' matches {} models: {}{}",
                hits.len(),
                names.join(", "),
                if hits.len() > 8 { ", ..." } else { "" }
            ))
        }
    }
}

pub fn run(
    m: &Machine,
    fit: &Fit,
    kv: KvPrecision,
    model: &str,
    ctx: Option<u32>,
    quant_filter: Option<&str>,
    target_tps: Option<f64>,
) -> i32 {
    let specs = catalog::load();
    let spec = match resolve(&specs, model) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let ctx = ctx.unwrap_or(DEFAULT_CTX);
    let tps = target_tps.unwrap_or(DEFAULT_TPS);

    // A context past what the model was trained for is not a plan, it is a
    // number. Say so rather than sizing memory for it.
    if let Some(trained) = spec.n_ctx_train.filter(|t| ctx > *t) {
        eprintln!(
            "{} was trained for {trained} tokens of context; {ctx} is past that.",
            spec.id
        );
        return 2;
    }

    let quants: Vec<&zc_model::Quant> = spec
        .quants
        .iter()
        .filter(|q| {
            quant_filter.is_none_or(|f| q.name.eq_ignore_ascii_case(f))
        })
        .collect();
    if quants.is_empty() {
        eprintln!(
            "{} has no quantisation named '{}'. It has: {}",
            spec.id,
            quant_filter.unwrap_or(""),
            spec.quants
                .iter()
                .map(|q| q.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        return 1;
    }

    let quants_first: &zc_model::Quant = quants[0];
    let mut o = String::new();
    o.push_str(&format!(
        "== plan ==  {} at {} context, KV {}, target {:.0} tok/s\n\n",
        spec.id,
        if ctx >= 1024 {
            format!("{}K", ctx / 1024)
        } else {
            ctx.to_string()
        },
        kv.tag().to_uppercase(),
        tps,
    ));
    o.push_str(&format!(
        "  this machine   {:.2} GiB budget, {:.0} GB/s measured, {}\n\n",
        gib(m.budget_idle),
        m.hw.ram_bw_gbs,
        zc_report::text::backend_label(m.backend),
    ));
    o.push_str(&format!(
        "  {:<8} {:>8} {:>7} {:>8}  {:>12}   {}\n",
        "quant", "weights", "KV", "total", "needs", "on this machine"
    ));

    for q in quants {
        let req = predict::requirement(spec, q, ctx, kv, UBATCH);
        let coef = predict::plan_eta(fit, m.backend, q);
        let need_bw = predict::required_bandwidth_gbs(spec, q, tps, coef.eta);
        let p = predict::predict_with(spec, q, &m.hw, kv, ctx.min(2048), UBATCH, fit);
        // Fit is judged against the requested context, which is the question
        // asked -- not against whatever context `predict` chose to operate at.
        let verdict = if req.total <= m.budget_idle {
            format!("fits, {:.0}-{:.0} t/s", p.decode_tok_s.0, p.decode_tok_s.1)
        } else {
            format!("over by {:.2} GiB", gib(req.total - m.budget_idle))
        };
        o.push_str(&format!(
            "  {:<8} {:>8.2} {:>7.2} {:>8.2}  {:>7.0} GB/s   {}\n",
            q.name,
            gib(req.weights),
            gib(req.kv),
            gib(req.total),
            need_bw,
            verdict,
        ));
    }

    // One coefficient for the footer, from a quantisation the model actually
    // has -- the family is what selects the bucket, and the footer is about
    // where the number came from.
    let shown = predict::plan_eta(fit, m.backend, quants_first);
    o.push_str("\n  GiB is memory, however it is provided -- RAM, VRAM or unified.\n");
    o.push_str(&format!(
        "  'needs' is the bandwidth for {:.0} tok/s at eta {:.3}, {} confidence.\n",
        tps,
        shown.eta,
        shown.confidence.label()
    ));
    o.push_str("  Bandwidth is checkable against a spec sheet. A GPU model name\n");
    o.push_str("  would be a lookup, and this tool puts no lookup under a number.\n");
    print!("{}", zc_report::text::block(&o));
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_accepts_plain_and_k_suffixed_numbers() {
        assert_eq!(parse_ctx("4096"), Some(4096));
        assert_eq!(parse_ctx("32k"), Some(32768));
        assert_eq!(parse_ctx("32K"), Some(32768));
        assert_eq!(parse_ctx(" 8K "), Some(8192));
        assert_eq!(parse_ctx("0"), None);
        assert_eq!(parse_ctx("K"), None);
        assert_eq!(parse_ctx("-1"), None);
        assert_eq!(parse_ctx("abc"), None);
        // Would overflow u32 once multiplied.
        assert_eq!(parse_ctx("999999999K"), None);
    }

    fn spec(id: &str) -> zc_model::ModelSpec {
        zc_model::ModelSpec {
            id: id.into(),
            n_layers: 1,
            n_embd: 1,
            n_vocab: 1,
            params: 1,
            attention: zc_model::spec::Attention::Gqa {
                n_kv_heads: 1,
                head_dim: 1,
            },
            moe: None,
            n_ctx_train: None,
            quants: vec![],
        }
    }

    /// An ambiguous prefix is refused. Planning for the wrong model in silence
    /// is worse than asking again.
    #[test]
    fn an_ambiguous_prefix_is_refused_not_guessed() {
        let specs = vec![spec("qwen3-8b"), spec("qwen3-4b"), spec("llama-3.1-8b")];
        assert_eq!(resolve(&specs, "qwen3-8b").unwrap().id, "qwen3-8b");
        assert_eq!(resolve(&specs, "llama").unwrap().id, "llama-3.1-8b");
        let err = resolve(&specs, "qwen3").unwrap_err();
        assert!(err.contains("matches 2"), "{err}");
        assert!(resolve(&specs, "nothing-like-this").is_err());
    }

    /// An exact id wins over a prefix that would otherwise be ambiguous.
    #[test]
    fn an_exact_id_beats_an_ambiguous_prefix() {
        let specs = vec![spec("qwen3-8b"), spec("qwen3-8b-instruct")];
        assert_eq!(resolve(&specs, "qwen3-8b").unwrap().id, "qwen3-8b");
    }
}
