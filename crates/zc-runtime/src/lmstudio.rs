//! LM Studio.
//!
//! Its OpenAI-compatible `/v1` surface reports token counts and no timings,
//! which would disqualify it. Its **native** `/api/v0` surface returns a
//! `stats` block — `tokens_per_second` and `time_to_first_token` — measured by
//! the runtime itself. That is the split we need, so this client speaks the
//! native API exclusively and LM Studio is calibration-grade.
//!
//! Like llama.cpp it reports no attention geometry, so the model spec comes
//! from the catalog via `catalog_match`.

use crate::{Endpoint, InstalledModel, RunStats, Runtime};
use zc_model::json;
use zc_model::json::escape;

const DEFAULT_PORT: u16 = 1234;

pub struct LmStudio {
    ep: Endpoint,
}

impl LmStudio {
    pub fn at(ep: Endpoint) -> Option<Self> {
        ep.responds("/api/v0/models").then_some(LmStudio { ep })
    }

    pub fn detect() -> Option<Self> {
        Self::at(Endpoint::from_env("LMSTUDIO_HOST", DEFAULT_PORT))
    }
}

/// Parse `/api/v0/models`.
///
/// Embedding models are filtered out: they are listed alongside chat models and
/// cannot be asked to generate, so offering one as a calibration target is a
/// guaranteed failed run.
pub fn parse_models(body: &str) -> Vec<InstalledModel> {
    let mut out = Vec::new();
    for obj in json::array_objects(body, "data") {
        let Some(id) = json::string(obj, "id") else {
            continue;
        };
        if json::string(obj, "type").is_some_and(|t| t != "llm" && t != "vlm") {
            continue;
        }
        out.push(InstalledModel {
            name: id,
            // LM Studio does not report file size. `catalog_match` then falls
            // back to the catalog's own byte count for this quantisation,
            // which is the only remaining source.
            size_bytes: 0,
            quant: json::string(obj, "quantization").unwrap_or_else(|| "unknown".into()),
            ..Default::default()
        });
    }
    out
}

/// Turn an `/api/v0/chat/completions` response into rates.
///
/// `tokens_per_second` is the decode rate directly. Prefill has to be derived:
/// `time_to_first_token` covers tokenisation plus the prompt pass, which for
/// the ~1200-token measurement prompt is prefill-dominated. It is the runtime's
/// own instrumentation, not our wall clock, which is what makes it usable.
pub fn parse_completion(body: &str, model: &str) -> RunStats {
    let stats = json::object_at(body, "stats").unwrap_or(body);
    let usage = json::object_at(body, "usage").unwrap_or(body);
    let prompt_tokens = json::number(usage, "prompt_tokens").unwrap_or(0.0);
    let ttft = json::number(stats, "time_to_first_token").unwrap_or(0.0);
    RunStats {
        model: model.to_string(),
        prompt_tokens: prompt_tokens as u32,
        eval_tokens: json::number(usage, "completion_tokens").unwrap_or(0.0) as u32,
        prefill_tok_s: if ttft > 0.0 { prompt_tokens / ttft } else { 0.0 },
        decode_tok_s: json::number(stats, "tokens_per_second").unwrap_or(0.0),
        // Loading happens on demand and is not broken out of the response.
        load_s: 0.0,
        // Context is fixed when the model is loaded, and reported back per
        // request. Recording our own request instead would be a fiction.
        n_ctx: json::object_at(body, "model_info")
            .and_then(|m| json::number(m, "context_length"))
            .map(|v| v as u32),
    }
}

impl Runtime for LmStudio {
    fn name(&self) -> &'static str {
        "lmstudio"
    }

    fn endpoint(&self) -> &Endpoint {
        &self.ep
    }

    fn list(&self) -> std::io::Result<Vec<InstalledModel>> {
        Ok(parse_models(&self.ep.get("/api/v0/models")?.body))
    }

    fn generate(
        &self,
        model: &str,
        num_predict: u32,
        _num_ctx: u32,
        nonce: u64,
    ) -> std::io::Result<RunStats> {
        let body = format!(
            r#"{{"model":"{}","messages":[{{"role":"user","content":"{}"}}],"temperature":0,"max_tokens":{},"stream":false}}"#,
            escape(model),
            escape(&crate::measurement_prompt(nonce)),
            num_predict
        );
        let r = self.ep.post("/api/v0/chat/completions", &body)?;
        if r.status != 200 {
            return Err(std::io::Error::other(format!(
                "lm studio returned {}: {}",
                r.status,
                r.body.chars().take(200).collect::<String>()
            )));
        }
        Ok(parse_completion(&r.body, model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Numbers taken from LM Studio's own documented response.
    /// 24 prompt tokens in 0.111 s is 216.2 tok/s of prefill.
    #[test]
    fn stats_block_supplies_both_halves_of_the_split() {
        let body = r#"{"id":"chatcmpl-1","object":"chat.completion","model":"granite-3.0-2b-instruct",
            "usage":{"prompt_tokens":24,"completion_tokens":53,"total_tokens":77},
            "stats":{"tokens_per_second":51.43709529007664,"time_to_first_token":0.111,
            "generation_time":0.954,"stop_reason":"eosFound"},
            "model_info":{"arch":"granite","quant":"Q4_K_M","format":"gguf","context_length":4096}}"#;
        let r = parse_completion(body, "granite-3.0-2b-instruct");
        assert_eq!(r.eval_tokens, 53);
        assert_eq!(r.prompt_tokens, 24);
        assert!((r.decode_tok_s - 51.437).abs() < 0.01, "{}", r.decode_tok_s);
        assert!((r.prefill_tok_s - 216.216).abs() < 0.01, "{}", r.prefill_tok_s);
        // The loaded context, not the one we asked for.
        assert_eq!(r.n_ctx, Some(4096));
    }

    #[test]
    fn a_response_without_stats_yields_zero_not_nan() {
        let r = parse_completion(r#"{"usage":{"prompt_tokens":10}}"#, "m");
        assert_eq!(r.decode_tok_s, 0.0);
        assert!(r.prefill_tok_s.is_finite());
    }

    /// An embedding model cannot generate, so offering one as a calibration
    /// target guarantees a failed run.
    #[test]
    fn embedding_models_are_not_offered_as_targets() {
        let body = r#"{"data":[
            {"id":"qwen3-4b","type":"llm","quantization":"Q4_K_M","max_context_length":32768},
            {"id":"nomic-embed-text","type":"embeddings","quantization":"F16"}]}"#;
        let v = parse_models(body);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].name, "qwen3-4b");
        assert_eq!(v[0].quant, "Q4_K_M");
    }
}
