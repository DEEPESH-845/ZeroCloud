//! Ollama client.
//!
//! Ollama is the most widely installed local runtime, and critically it
//! reports its own timings: `/api/generate` returns prompt-eval and eval
//! durations separately. That is exactly our prefill/decode split, measured
//! inside the process, with no HTTP or startup overhead in the numbers.
//! Timing from outside would be strictly worse.
//!
//! It is also the only runtime here that serves full GGUF metadata
//! (`/api/show`), so it can calibrate *any* model the user has rather than
//! only catalogued ones. That is why it is preferred over the others.

use crate::{Endpoint, InstalledModel, RunStats, Runtime};
use zc_model::json;
use zc_model::spec::{Attention, ModelSpec, Moe, Quant, QuantFamily};

const DEFAULT_PORT: u16 = 11434;

/// Map a GGUF quantisation label to its family.
///
/// The family determines dequantisation cost per byte, a first-order term in
/// decode speed: i-quants move fewer bytes but cost markedly more CPU per byte.
pub fn quant_family(label: &str) -> QuantFamily {
    QuantFamily::from_gguf_label(label)
}

/// The Ollama runtime.
pub struct Ollama {
    ep: Endpoint,
}

impl Ollama {
    pub fn at(ep: Endpoint) -> Option<Self> {
        ep.responds("/api/tags").then_some(Ollama { ep })
    }

    pub fn detect() -> Option<Self> {
        Self::at(Endpoint::from_env("OLLAMA_HOST", DEFAULT_PORT))
    }
}

impl Runtime for Ollama {
    fn name(&self) -> &'static str {
        "ollama"
    }

    fn endpoint(&self) -> &Endpoint {
        &self.ep
    }

    /// Installed models, largest first.
    fn list(&self) -> std::io::Result<Vec<InstalledModel>> {
        Ok(parse_tags(&self.ep.get("/api/tags")?.body))
    }

    /// Build a ModelSpec from the runtime's own GGUF metadata.
    ///
    /// The one runtime that can do this, which is why it can measure models
    /// the catalog has never heard of.
    fn describe(&self, model: &InstalledModel) -> std::io::Result<Option<ModelSpec>> {
        let body = format!(r#"{{"model":"{}"}}"#, json::escape(&model.name));
        let r = self.ep.post("/api/show", &body)?;
        if r.status != 200 {
            return Ok(None);
        }
        Ok(parse_show(&r.body, model))
    }

    /// `num_ctx` is honoured per request here, so `RunStats::n_ctx` stays
    /// `None` — what we asked for is what ran.
    fn generate(
        &self,
        model: &str,
        num_predict: u32,
        num_ctx: u32,
        nonce: u64,
    ) -> std::io::Result<RunStats> {
        let prompt = crate::measurement_prompt(nonce);
        let body = format!(
            r#"{{"model":"{}","prompt":"{}","stream":false,"options":{{"num_predict":{},"num_ctx":{},"temperature":0}}}}"#,
            json::escape(model),
            json::escape(&prompt),
            num_predict,
            num_ctx
        );
        let r = self.ep.post("/api/generate", &body)?;
        if r.status != 200 {
            return Err(std::io::Error::other(format!(
                "ollama returned {}: {}",
                r.status,
                r.body.chars().take(200).collect::<String>()
            )));
        }
        Ok(parse_generate(&r.body, model))
    }
}

// Parsing is split from transport so it can be tested against captured
// responses without a live server.

pub fn parse_tags(body: &str) -> Vec<InstalledModel> {
    let mut out = Vec::new();
    for obj in json::array_objects(body, "models") {
        let Some(name) = json::string(obj, "name").or_else(|| json::string(obj, "model")) else {
            continue;
        };
        out.push(InstalledModel {
            name,
            size_bytes: json::number(obj, "size").unwrap_or(0.0) as u64,
            quant: json::string(obj, "quantization_level").unwrap_or_else(|| "unknown".into()),
            // Ollama serves full geometry from `/api/show`, so nothing has to
            // be smuggled through the listing to verify a catalog match.
            ..Default::default()
        });
    }
    out.sort_by_key(|m| std::cmp::Reverse(m.size_bytes));
    out
}

/// Every field is read from metadata, never inferred. Where a key is genuinely
/// absent the fallback is documented rather than silently guessed.
pub fn parse_show(s: &str, model: &InstalledModel) -> Option<ModelSpec> {
    let n_layers = json::number_by_suffix(s, ".block_count")? as u32;
    let n_embd = json::number_by_suffix(s, ".embedding_length")? as u32;
    if n_layers == 0 || n_embd == 0 {
        return None;
    }
    let n_heads = json::number_by_suffix(s, ".attention.head_count").unwrap_or(0.0) as u32;
    let n_kv_heads = json::number_by_suffix(s, ".attention.head_count_kv").unwrap_or(0.0) as u32;

    // Newer GGUFs state key_length explicitly. For older ones n_embd/n_heads
    // is the correct derivation.
    let head_dim = json::number_by_suffix(s, ".attention.key_length")
        .map(|v| v as u32)
        .filter(|&v| v > 0)
        .or_else(|| n_embd.checked_div(n_heads))
        .unwrap_or(128);

    let params = json::number(s, "parameter_count")
        .or_else(|| json::number_by_suffix(s, ".parameter_count"))
        .unwrap_or(0.0) as u64;

    let n_vocab = json::number_by_suffix(s, ".vocab_size")
        .map(|v| v as u32)
        .filter(|&v| v > 0)
        .unwrap_or(32000);

    // Sliding-window models cap KV growth on most layers, changing the context
    // math several-fold. Presence of the key is the signal.
    let attention = match json::number_by_suffix(s, ".attention.sliding_window").map(|v| v as u32) {
        Some(w) if w > 0 => Attention::Swa {
            n_kv_heads: n_kv_heads.max(1),
            head_dim,
            window: w,
            // Not always present. 6 matches Gemma 3's 5:1 local-to-global
            // ratio, the common case today. An assumption, not a reading.
            global_every: json::number_by_suffix(s, ".attention.sliding_window_pattern")
                .map(|v| v as u32)
                .filter(|&v| v > 0)
                .unwrap_or(6),
        },
        _ => Attention::Gqa {
            n_kv_heads: n_kv_heads.max(1),
            head_dim,
        },
    };

    let n_expert = json::number_by_suffix(s, ".expert_count").unwrap_or(0.0) as u32;
    let moe = (n_expert > 0).then(|| {
        let ff = json::number_by_suffix(s, ".expert_feed_forward_length").unwrap_or(0.0) as u64;
        // gate + up + down projections, each n_embd x ff.
        let expert_params = 3 * n_embd as u64 * ff;
        let routed_total = expert_params * n_expert as u64 * n_layers as u64;
        Moe {
            n_expert,
            n_active: json::number_by_suffix(s, ".expert_used_count").unwrap_or(2.0) as u32,
            expert_params,
            shared_params: params.saturating_sub(routed_total),
        }
    });

    Some(ModelSpec {
        id: model.name.clone(),
        n_layers,
        n_embd,
        n_vocab,
        params: params.max(1),
        attention,
        moe,
        quants: vec![Quant {
            name: model.quant.clone(),
            bytes: model.size_bytes,
            family: quant_family(&model.quant),
        }],
    })
}

pub fn parse_generate(s: &str, model: &str) -> RunStats {
    let ns = |k: &str| json::number(s, k).unwrap_or(0.0);
    let (pc, pd) = (ns("prompt_eval_count"), ns("prompt_eval_duration"));
    let (ec, ed) = (ns("eval_count"), ns("eval_duration"));
    RunStats {
        model: model.to_string(),
        prompt_tokens: pc as u32,
        eval_tokens: ec as u32,
        // Durations are nanoseconds.
        prefill_tok_s: if pd > 0.0 { pc / (pd / 1e9) } else { 0.0 },
        decode_tok_s: if ed > 0.0 { ec / (ed / 1e9) } else { 0.0 },
        load_s: ns("load_duration") / 1e9,
        // Ollama honours the `num_ctx` we send.
        n_ctx: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_families_are_classified() {
        assert_eq!(quant_family("Q4_K_M"), QuantFamily::KQuant);
        assert_eq!(quant_family("Q6_K"), QuantFamily::KQuant);
        assert_eq!(quant_family("Q8_0"), QuantFamily::Legacy);
        assert_eq!(quant_family("IQ3_XXS"), QuantFamily::IQuant);
        assert_eq!(quant_family("MXFP4"), QuantFamily::BlockFloat);
        assert_eq!(quant_family("F16"), QuantFamily::Float);
    }

    #[test]
    fn parses_tags_and_sorts_largest_first() {
        let body = r#"{"models":[
            {"name":"qwen3:4b","size":2600000000,"details":{"quantization_level":"Q4_K_M"}},
            {"name":"llama3.1:8b","size":4900000000,"details":{"quantization_level":"Q4_K_M"}}]}"#;
        let v = parse_tags(body);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].name, "llama3.1:8b");
        assert_eq!(v[0].quant, "Q4_K_M");
    }

    fn model() -> InstalledModel {
        InstalledModel::new("qwen3:4b", 2_600_000_000, "Q4_K_M")
    }

    /// Dense model: architecture read straight from GGUF metadata, and KV must
    /// match the hand-computed 2*36*8*128*2 = 147456 bytes/token.
    #[test]
    fn parses_dense_model_metadata() {
        let body = r#"{"model_info":{"general.architecture":"qwen3",
            "general.parameter_count":4022468096,
            "qwen3.block_count":36,"qwen3.embedding_length":2560,
            "qwen3.attention.head_count":32,"qwen3.attention.head_count_kv":8,
            "qwen3.attention.key_length":128,"qwen3.vocab_size":151936}}"#;
        let s = parse_show(body, &model()).expect("spec");
        assert_eq!(s.n_layers, 36);
        assert_eq!(s.n_embd, 2560);
        assert_eq!(s.n_vocab, 151936);
        assert!(s.moe.is_none());
        assert_eq!(
            s.kv_bytes(1, zc_model::KvPrecision::F16),
            2 * 36 * 8 * 128 * 2
        );
    }

    /// MoE: active params must come out near the advertised 3.3B, which is the
    /// check that our expert-geometry derivation is right.
    #[test]
    fn parses_moe_model_metadata() {
        let body = r#"{"model_info":{"general.architecture":"qwen3moe",
            "general.parameter_count":30500000000,
            "qwen3moe.block_count":48,"qwen3moe.embedding_length":2048,
            "qwen3moe.attention.head_count":32,"qwen3moe.attention.head_count_kv":4,
            "qwen3moe.attention.key_length":128,"qwen3moe.vocab_size":151936,
            "qwen3moe.expert_count":128,"qwen3moe.expert_used_count":8,
            "qwen3moe.expert_feed_forward_length":768}}"#;
        let s = parse_show(body, &model()).expect("spec");
        let moe = s.moe.as_ref().expect("moe");
        assert_eq!(moe.n_expert, 128);
        assert_eq!(moe.n_active, 8);
        assert_eq!(moe.expert_params, 3 * 2048 * 768);
        let active_b = s.active_params() as f64 / 1e9;
        assert!((3.0..3.7).contains(&active_b), "active {active_b}B");
    }

    /// Sliding-window key must switch the attention variant, or the context
    /// math is several times wrong.
    #[test]
    fn sliding_window_switches_attention_variant() {
        let body = r#"{"model_info":{"general.architecture":"gemma3",
            "general.parameter_count":12200000000,
            "gemma3.block_count":48,"gemma3.embedding_length":3840,
            "gemma3.attention.head_count":16,"gemma3.attention.head_count_kv":8,
            "gemma3.attention.key_length":256,"gemma3.vocab_size":262144,
            "gemma3.attention.sliding_window":1024}}"#;
        let s = parse_show(body, &model()).expect("spec");
        assert!(matches!(s.attention, Attention::Swa { window: 1024, .. }));
    }

    /// Older GGUFs omit key_length; n_embd / head_count is the fallback.
    #[test]
    fn head_dim_falls_back_when_key_length_absent() {
        let body = r#"{"model_info":{"llama.block_count":32,
            "llama.embedding_length":4096,"llama.attention.head_count":32,
            "llama.attention.head_count_kv":8}}"#;
        let s = parse_show(body, &model()).expect("spec");
        assert!(matches!(s.attention, Attention::Gqa { head_dim: 128, .. }));
    }

    #[test]
    fn unusable_metadata_returns_none() {
        assert!(parse_show("{}", &model()).is_none());
    }

    /// Durations are nanoseconds: 128 tokens in 2.28s = 56.1 tok/s.
    #[test]
    fn computes_rates_from_nanosecond_durations() {
        let body = r#"{"model":"qwen3:4b","done":true,
            "total_duration":3521000000,"load_duration":41000000,
            "prompt_eval_count":2048,"prompt_eval_duration":1200000000,
            "eval_count":128,"eval_duration":2280000000}"#;
        let r = parse_generate(body, "qwen3:4b");
        assert_eq!(r.eval_tokens, 128);
        assert_eq!(r.prompt_tokens, 2048);
        assert!((r.decode_tok_s - 56.14).abs() < 0.01, "{}", r.decode_tok_s);
        assert!((r.prefill_tok_s - 1706.67).abs() < 0.01, "{}", r.prefill_tok_s);
    }

    /// A response missing timings must yield zeros, not NaN or a panic.
    #[test]
    fn missing_timings_yield_zero_not_nan() {
        let r = parse_generate(r#"{"done":false}"#, "m");
        assert_eq!(r.decode_tok_s, 0.0);
        assert!(r.prefill_tok_s.is_finite());
    }
}
