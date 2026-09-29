//! The one query layer behind `zc serve` and `zc serve --mcp`.
//!
//! Both take string key/value pairs -- a URL query or an MCP tool's
//! arguments -- and answer with exactly the JSON `zc check --json` and
//! `zc plan --json` print. Nothing here renders; it validates parameters and
//! calls the functions the CLI calls, so a server cannot drift from the
//! command it fronts.
//!
//! Unknown parameters are refused, for the reason `main.rs` refuses unknown
//! flags: `?top=5&kv=q8&al_quants=true` must not quietly answer a different
//! question from the one asked.

use crate::{check, machine::Machine, plan_cmd};
use zc_model::{catalog, Fit, KvPrecision};

/// A failed query: HTTP status, and the message both transports report.
pub type Failure = (u16, String);

pub const CHECK_PARAMS: &[&str] = &["model", "top", "all_quants", "kv"];
pub const PLAN_PARAMS: &[&str] = &["model", "context", "quant", "kv", "target_tps"];

fn get<'a>(p: &'a [(String, String)], key: &str) -> Option<&'a str> {
    p.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

fn bad(msg: String) -> Failure {
    (400, msg)
}

fn only(p: &[(String, String)], allowed: &[&str]) -> Result<(), Failure> {
    match p.iter().find(|(k, _)| !allowed.contains(&k.as_str())) {
        Some((k, _)) => Err(bad(format!(
            "unknown parameter '{k}' (accepted: {})",
            allowed.join(", ")
        ))),
        None => Ok(()),
    }
}

fn kv(p: &[(String, String)]) -> Result<KvPrecision, Failure> {
    match get(p, "kv") {
        None => Ok(KvPrecision::DEFAULT),
        Some(v) => KvPrecision::parse(v)
            .ok_or_else(|| bad(format!("unknown kv '{v}' (expected f16, q8 or q4)"))),
    }
}

fn flag(p: &[(String, String)], key: &str) -> Result<bool, Failure> {
    match get(p, key) {
        None | Some("false" | "0") => Ok(false),
        Some("true" | "1" | "") => Ok(true),
        Some(v) => Err(bad(format!("{key} is true or false, got '{v}'"))),
    }
}

/// `zc check --json`, optionally narrowed to models whose id contains `model`.
pub fn check(m: &Machine, fit: &Fit, p: &[(String, String)]) -> Result<String, Failure> {
    only(p, CHECK_PARAMS)?;
    let kv = kv(p)?;
    let all = flag(p, "all_quants")?;
    // Same rule as the CLI: all_quants lifts the row limit unless one is given.
    let top = match get(p, "top") {
        Some(v) => match v.parse::<usize>() {
            Ok(n) if n > 0 => Some(n),
            _ => return Err(bad(format!("top is a positive integer, got '{v}'"))),
        },
        None if all => None,
        None => Some(check::DEFAULT_TOP),
    };
    let specs = catalog::load();
    let (mut rows, _) = check::rows(&specs, m, fit, kv, all);
    if let Some(want) = get(p, "model") {
        let want = want.to_ascii_lowercase();
        rows.retain(|r| r.model_id.to_ascii_lowercase().contains(&want));
        if rows.is_empty() {
            return Err((404, format!("no catalog model matches '{want}'")));
        }
    }
    let total = rows.len();
    if let Some(n) = top {
        rows.truncate(n);
    }
    Ok(zc_report::json::render(&check::report(m, fit, kv, rows, total)))
}

/// `zc plan --json`. A missing model or quantisation is 404; a malformed
/// argument or a context past training is 400.
pub fn plan(m: &Machine, fit: &Fit, p: &[(String, String)]) -> Result<String, Failure> {
    only(p, PLAN_PARAMS)?;
    let kv = kv(p)?;
    let model = get(p, "model")
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| bad("plan needs a model, e.g. model=qwen3-8b".into()))?;
    let ctx = match get(p, "context") {
        None => None,
        Some(v) => Some(plan_cmd::parse_ctx(v).ok_or_else(|| {
            bad(format!("context is a token count like 4096 or 32K, got '{v}'"))
        })?),
    };
    let tps = match get(p, "target_tps") {
        None => None,
        Some(v) => match v.parse::<f64>() {
            Ok(t) if t > 0.0 && t.is_finite() => Some(t),
            _ => return Err(bad(format!("target_tps is a positive number, got '{v}'"))),
        },
    };
    match plan_cmd::build(m, fit, kv, model, ctx, get(p, "quant"), tps) {
        Ok(plan) => Ok(plan_cmd::json(m, &plan)),
        Err((1, msg)) => Err((404, msg)),
        Err((_, msg)) => Err((400, msg)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    /// A misspelt parameter must fail, not answer the default question.
    #[test]
    fn unknown_parameters_are_refused() {
        assert!(only(&p(&[("top", "5")]), CHECK_PARAMS).is_ok());
        let e = only(&p(&[("al_quants", "true")]), CHECK_PARAMS).unwrap_err();
        assert_eq!(e.0, 400);
        assert!(e.1.contains("al_quants"));
        assert!(only(&p(&[("top", "5")]), PLAN_PARAMS).is_err());
    }

    #[test]
    fn flags_and_kv_parse_strictly() {
        assert_eq!(flag(&p(&[]), "all_quants"), Ok(false));
        assert_eq!(flag(&p(&[("all_quants", "true")]), "all_quants"), Ok(true));
        assert_eq!(flag(&p(&[("all_quants", "")]), "all_quants"), Ok(true));
        assert!(flag(&p(&[("all_quants", "yes")]), "all_quants").is_err());
        assert!(kv(&p(&[("kv", "q8")])).is_ok());
        assert_eq!(kv(&p(&[("kv", "q5")])).unwrap_err().0, 400);
    }
}
