//! llama.cpp's `llama-server`.
//!
//! Calibration-grade: `/completion` returns a `timings` object with
//! `prompt_ms` and `predicted_ms` measured inside the process, which is exactly
//! the prefill/decode split, with no HTTP or scheduling overhead in the
//! numbers.
//!
//! It is identified through `/props` rather than `/v1/models`, and that
//! ordering is load-bearing: MLX's server also defaults to port 8080 and also
//! answers `/v1/models`, so a positive identification has to come first or the
//! two are indistinguishable.
//!
//! What it does *not* give is attention geometry — `/v1/models` reports
//! `n_params`, `n_embd`, `n_vocab` and `size`, but no layer count, no KV head
//! count, no sliding window. Those cannot be inferred from a filename, so the
//! spec comes from the catalog via `catalog_match`, with everything llama.cpp
//! did report used to *verify* the match rather than trust it.

use crate::openai::parse_models;
use crate::{Endpoint, InstalledModel, RunStats, Runtime};
use zc_model::json;
use zc_model::json::escape;

/// `llama-server --port` defaults here, and so does MLX.
const DEFAULT_PORT: u16 = 8080;

pub struct LlamaCpp {
    ep: Endpoint,
    /// Context the server was launched with. Fixed at startup — llama.cpp has
    /// no per-request context setting, so recording what we asked for instead
    /// of what it runs would put a wrong number in the shared dataset.
    n_ctx: Option<u32>,
}

impl LlamaCpp {
    pub fn at(ep: Endpoint) -> Option<Self> {
        let r = ep.get("/props").ok()?;
        if r.status != 200 || !is_llama_cpp(&r.body) {
            return None;
        }
        Some(LlamaCpp {
            n_ctx: server_ctx(&r.body),
            ep,
        })
    }

    pub fn detect() -> Option<Self> {
        Self::at(Endpoint::from_env("LLAMA_SERVER_HOST", DEFAULT_PORT))
    }
}

/// Positive identification, not merely "something answered".
///
/// `build_info` and `model_path` are llama.cpp's own fields; nothing else
/// serving this port emits them.
fn is_llama_cpp(props: &str) -> bool {
    json::string(props, "build_info").is_some()
        || json::string(props, "model_path").is_some()
        || json::object_at(props, "default_generation_settings").is_some()
}

fn server_ctx(props: &str) -> Option<u32> {
    let settings = json::object_at(props, "default_generation_settings")?;
    json::number(settings, "n_ctx")
        .map(|v| v as u32)
        .filter(|&v| v > 0)
}

/// Turn a `/completion` response into rates.
///
/// Durations are milliseconds here, unlike Ollama's nanoseconds — a units
/// mix-up would be off by a factor of a million and still look like a plausible
/// number, so the conversion lives in one place with a test.
pub fn parse_completion(body: &str, model: &str, n_ctx: Option<u32>) -> RunStats {
    let t = json::object_at(body, "timings").unwrap_or(body);
    let n = |k: &str| json::number(t, k).unwrap_or(0.0);
    let (pn, pms) = (n("prompt_n"), n("prompt_ms"));
    let (en, ems) = (n("predicted_n"), n("predicted_ms"));
    RunStats {
        model: model.to_string(),
        prompt_tokens: pn as u32,
        eval_tokens: en as u32,
        prefill_tok_s: if pms > 0.0 { pn / (pms / 1e3) } else { 0.0 },
        decode_tok_s: if ems > 0.0 { en / (ems / 1e3) } else { 0.0 },
        // The server holds the model resident; there is no per-request load.
        load_s: 0.0,
        n_ctx,
    }
}

impl Runtime for LlamaCpp {
    fn name(&self) -> &'static str {
        "llamacpp"
    }

    fn endpoint(&self) -> &Endpoint {
        &self.ep
    }

    fn list(&self) -> std::io::Result<Vec<InstalledModel>> {
        Ok(parse_models(&self.ep.get("/v1/models")?.body))
    }

    fn generate(
        &self,
        model: &str,
        num_predict: u32,
        _num_ctx: u32,
        nonce: u64,
    ) -> std::io::Result<RunStats> {
        // `cache_prompt` defaults to true. The nonce already defeats prefix
        // reuse, but a cached prefill reports a near-zero prompt_ms and would
        // silently corrupt the prefill coefficient, so say it twice.
        let body = format!(
            r#"{{"prompt":"{}","n_predict":{},"temperature":0,"cache_prompt":false,"stream":false}}"#,
            escape(&crate::measurement_prompt(nonce)),
            num_predict
        );
        let r = self.ep.post("/completion", &body)?;
        if r.status != 200 {
            return Err(std::io::Error::other(format!(
                "llama-server returned {}: {}",
                r.status,
                r.body.chars().take(200).collect::<String>()
            )));
        }
        Ok(parse_completion(&r.body, model, self.n_ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Milliseconds, not nanoseconds. 35 tokens in 661.064 ms is 52.94 tok/s,
    /// and one prompt token in 30.958 ms is 32.30 tok/s — both taken straight
    /// from the server's documented example.
    #[test]
    fn timings_are_milliseconds_not_nanoseconds() {
        let body = r#"{"content":"x","timings":{"cache_n":236,"prompt_n":1,
            "prompt_ms":30.958,"prompt_per_token_ms":30.958,
            "predicted_n":35,"predicted_ms":661.064,
            "predicted_per_second":52.94494935437416}}"#;
        let r = parse_completion(body, "m", Some(4096));
        assert_eq!(r.eval_tokens, 35);
        assert!((r.decode_tok_s - 52.945).abs() < 0.01, "{}", r.decode_tok_s);
        assert!((r.prefill_tok_s - 32.302).abs() < 0.01, "{}", r.prefill_tok_s);
        assert_eq!(r.n_ctx, Some(4096));
    }

    #[test]
    fn missing_timings_yield_zero_not_nan() {
        let r = parse_completion(r#"{"content":""}"#, "m", None);
        assert_eq!(r.decode_tok_s, 0.0);
        assert!(r.prefill_tok_s.is_finite());
    }

    /// MLX answers on the same port. Anything that cannot prove it is
    /// llama.cpp must be left for the OpenAI-compatible probe, which knows it
    /// cannot calibrate.
    #[test]
    fn only_llama_cpp_identifies_as_llama_cpp() {
        assert!(is_llama_cpp(r#"{"build_info":"b4321-abc","total_slots":1}"#));
        assert!(is_llama_cpp(
            r#"{"model_path":"/m/llama-3.1-8b-Q4_K_M.gguf"}"#
        ));
        assert!(!is_llama_cpp(r#"{"object":"list","data":[]}"#));
        assert!(!is_llama_cpp("{}"));
    }

    /// The server fixes context at launch, so this is the only truthful figure
    /// for the record.
    #[test]
    fn context_comes_from_the_server_not_the_request() {
        let props = r#"{"default_generation_settings":{"id":0,"n_ctx":8192,
            "params":{"n_predict":-1}},"total_slots":1,"build_info":"b1"}"#;
        assert_eq!(server_ctx(props), Some(8192));
        assert_eq!(server_ctx(r#"{"build_info":"b1"}"#), None);
    }
}
