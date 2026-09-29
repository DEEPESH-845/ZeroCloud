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

/// One quantisation's requirement, and what this machine makes of it.
pub struct PlanRow {
    pub quant: String,
    pub weights: u64,
    pub kv: u64,
    pub total: u64,
    /// GB/s needed for the target rate, by inverting the decode model.
    pub needs_gbs: f64,
    /// Judged against the requested context, which is the question asked.
    pub fits: bool,
    pub decode_tok_s: (f64, f64),
}

pub struct Plan {
    pub model_id: String,
    pub ctx: u32,
    pub kv: KvPrecision,
    pub target_tps: f64,
    pub eta: f64,
    pub confidence: &'static str,
    pub rows: Vec<PlanRow>,
}

/// Compute a plan, or the exit code and message `zc plan` would fail with:
/// 1 for no such model or quantisation, 2 for a context past training.
pub fn build(
    m: &Machine,
    fit: &Fit,
    kv: KvPrecision,
    model: &str,
    ctx: Option<u32>,
    quant_filter: Option<&str>,
    target_tps: Option<f64>,
) -> Result<Plan, (i32, String)> {
    let specs = catalog::load();
    let spec = resolve(&specs, model).map_err(|e| (1, e))?;
    let ctx = ctx.unwrap_or(DEFAULT_CTX);
    let tps = target_tps.unwrap_or(DEFAULT_TPS);

    // A context past what the model was trained for is not a plan, it is a
    // number. Say so rather than sizing memory for it.
    if let Some(trained) = spec.n_ctx_train.filter(|t| ctx > *t) {
        return Err((
            2,
            format!("{} was trained for {trained} tokens of context; {ctx} is past that.", spec.id),
        ));
    }

    let quants: Vec<&zc_model::Quant> = spec
        .quants
        .iter()
        .filter(|q| quant_filter.is_none_or(|f| q.name.eq_ignore_ascii_case(f)))
        .collect();
    if quants.is_empty() {
        return Err((
            1,
            format!(
                "{} has no quantisation named '{}'. It has: {}",
                spec.id,
                quant_filter.unwrap_or(""),
                spec.quants.iter().map(|q| q.name.as_str()).collect::<Vec<_>>().join(", ")
            ),
        ));
    }

    let rows = quants
        .iter()
        .map(|&q| {
            let req = predict::requirement(spec, q, ctx, kv, UBATCH);
            let coef = predict::plan_eta(fit, m.backend, q);
            let p = predict::predict_with(spec, q, &m.hw, kv, ctx.min(2048), UBATCH, fit);
            PlanRow {
                quant: q.name.clone(),
                weights: req.weights,
                kv: req.kv,
                total: req.total,
                needs_gbs: predict::required_bandwidth_gbs(spec, q, tps, coef.eta),
                fits: req.total <= m.budget_idle,
                decode_tok_s: p.decode_tok_s,
            }
        })
        .collect();

    // One coefficient for the footer, from a quantisation the model actually
    // has -- the family is what selects the bucket, and the footer is about
    // where the number came from.
    let shown = predict::plan_eta(fit, m.backend, quants[0]);
    Ok(Plan {
        model_id: spec.id.clone(),
        ctx,
        kv,
        target_tps: tps,
        eta: shown.eta,
        confidence: shown.confidence.label(),
        rows,
    })
}

pub fn text(m: &Machine, plan: &Plan) -> String {
    let mut o = String::new();
    o.push_str(&format!(
        "== plan ==  {} at {} context, KV {}, target {:.0} tok/s\n\n",
        plan.model_id,
        if plan.ctx >= 1024 {
            format!("{}K", plan.ctx / 1024)
        } else {
            plan.ctx.to_string()
        },
        plan.kv.tag().to_uppercase(),
        plan.target_tps,
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
    for r in &plan.rows {
        let verdict = if r.fits {
            format!("fits, {:.0}-{:.0} t/s", r.decode_tok_s.0, r.decode_tok_s.1)
        } else {
            format!("over by {:.2} GiB", gib(r.total - m.budget_idle))
        };
        o.push_str(&format!(
            "  {:<8} {:>8.2} {:>7.2} {:>8.2}  {:>7.0} GB/s   {}\n",
            r.quant,
            gib(r.weights),
            gib(r.kv),
            gib(r.total),
            r.needs_gbs,
            verdict,
        ));
    }
    o.push_str("\n  GiB is memory, however it is provided -- RAM, VRAM or unified.\n");
    o.push_str(&format!(
        "  'needs' is the bandwidth for {:.0} tok/s at eta {:.3}, {} confidence.\n",
        plan.target_tps, plan.eta, plan.confidence
    ));
    o.push_str("  Bandwidth is checkable against a spec sheet. A GPU model name\n");
    o.push_str("  would be a lookup, and this tool puts no lookup under a number.\n");
    zc_report::text::block(&o)
}

/// The same plan for a script or an agent. Bytes, not GiB; `decode_tok_s` is
/// `null` for a quantisation that does not fit, because a speed for a model
/// that cannot load is not a claim anyone can check.
pub fn json(m: &Machine, plan: &Plan) -> String {
    use zc_model::json::escape;
    let rows: Vec<String> = plan
        .rows
        .iter()
        .map(|r| {
            format!(
                "{{\"quant\":\"{}\",\"weights_bytes\":{},\"kv_bytes\":{},\"total_bytes\":{},\
                 \"needs_gbs\":{},\"fits\":{},\"decode_tok_s\":{}}}",
                escape(&r.quant),
                r.weights,
                r.kv,
                r.total,
                num(r.needs_gbs),
                r.fits,
                if r.fits {
                    format!("{{\"low\":{},\"high\":{}}}", num(r.decode_tok_s.0), num(r.decode_tok_s.1))
                } else {
                    "null".into()
                },
            )
        })
        .collect();
    format!(
        "{{\"model\":\"{}\",\"context\":{},\"kv_precision\":\"{}\",\"target_tok_s\":{},\
         \"eta\":{},\"confidence\":\"{}\",\"machine\":{{\"budget_bytes\":{},\"ram_bw_gbs\":{},\
         \"backend\":\"{}\"}},\"quants\":[{}]}}",
        escape(&plan.model_id),
        plan.ctx,
        plan.kv.tag(),
        num(plan.target_tps),
        num(plan.eta),
        plan.confidence,
        m.budget_idle,
        num(m.hw.ram_bw_gbs),
        zc_report::backend_tag(m.backend),
        rows.join(","),
    )
}

/// JSON has no NaN or Infinity; a non-finite number is `null`.
fn num(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.3}")
    } else {
        "null".into()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    m: &Machine,
    fit: &Fit,
    kv: KvPrecision,
    model: &str,
    ctx: Option<u32>,
    quant_filter: Option<&str>,
    target_tps: Option<f64>,
    as_json: bool,
) -> i32 {
    match build(m, fit, kv, model, ctx, quant_filter, target_tps) {
        Ok(plan) if as_json => {
            println!("{}", json(m, &plan));
            0
        }
        Ok(plan) => {
            print!("{}", text(m, &plan));
            0
        }
        Err((code, msg)) => {
            eprintln!("{msg}");
            code
        }
    }
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
