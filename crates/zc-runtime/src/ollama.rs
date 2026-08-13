//! Ollama client.
//!
//! Ollama is the most widely installed local runtime, and critically it
//! reports its own timings: `/api/generate` returns prompt-eval and eval
//! durations separately. That is exactly our prefill/decode split, measured
//! inside the process, with no HTTP or startup overhead in the numbers.
//! Timing from outside would be strictly worse.
//!
//! `/api/show` returns the model's GGUF metadata, so we can build a ModelSpec
//! for *any* model the user has rather than only catalog entries.

use crate::http;
use zc_model::json;
use zc_model::spec::{Attention, ModelSpec, Moe, Quant, QuantFamily};

/// Where the runtime lives.
///
/// Carried explicitly rather than hardcoded for two reasons: users genuinely
/// do run Ollama on another host or port (Ollama itself reads `OLLAMA_HOST`),
/// and a hardcoded endpoint makes the happy path impossible to test without
/// installing a runtime.
#[derive(Debug, Clone)]
pub struct Endpoint {
    pub host: String,
    pub port: u16,
}

impl Default for Endpoint {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 11434,
        }
    }
}

impl std::fmt::Display for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.host, self.port)
    }
}

/// Parse an `OLLAMA_HOST` value: `host`, `host:port`, or `http://host:port`.
///
/// Split from `from_env` so it can be tested without mutating process
/// environment, which is global state and races with parallel tests.
pub fn parse_host(raw: &str) -> Option<Endpoint> {
    let raw = raw
        .trim()
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_end_matches('/');
    if raw.is_empty() {
        return None;
    }
    // `[::1]:11434` — the bracketed form is the only unambiguous way to write
    // an IPv6 address with a port. Strip the brackets: the resolver wants the
    // bare address, and `[::1]` matches neither an IP literal nor a DNS name.
    if let Some(rest) = raw.strip_prefix('[') {
        let (h, tail) = rest.split_once(']')?;
        return Some(Endpoint {
            host: h.into(),
            port: match tail.strip_prefix(':') {
                Some(p) => p.parse().ok()?,
                None => 11434,
            },
        });
    }
    // A bare IPv6 address has several colons and no port; splitting on the last
    // one turned `::1` into host ":" port 1.
    if raw.matches(':').count() > 1 {
        return Some(Endpoint { host: raw.into(), port: 11434 });
    }
    Some(match raw.rsplit_once(':') {
        Some((h, p)) => Endpoint {
            host: if h.is_empty() { "127.0.0.1".into() } else { h.into() },
            port: p.parse().ok()?,
        },
        None => Endpoint {
            host: raw.into(),
            port: 11434,
        },
    })
}

#[derive(Debug, Clone)]
pub struct InstalledModel {
    pub name: String,
    /// On-disk size in bytes — the model's quantised weight size.
    pub size_bytes: u64,
    pub quant: String,
}

/// What the runtime measured, as opposed to what we predicted.
#[derive(Debug, Clone)]
pub struct RunStats {
    pub model: String,
    pub prompt_tokens: u32,
    pub eval_tokens: u32,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    pub load_s: f64,
}

/// Map a GGUF quantisation label to its family.
///
/// The family determines dequantisation cost per byte, a first-order term in
/// decode speed: i-quants move fewer bytes but cost markedly more CPU per byte.
pub fn quant_family(label: &str) -> QuantFamily {
    QuantFamily::from_gguf_label(label)
}

impl Endpoint {
    pub fn from_env() -> Self {
        std::env::var("OLLAMA_HOST")
            .ok()
            .and_then(|v| parse_host(&v))
            .unwrap_or_default()
    }

    pub fn is_running(&self) -> bool {
        http::get(&self.host, self.port, "/api/tags").is_ok_and(|r| r.status == 200)
    }

    /// Installed models, largest first.
    pub fn list(&self) -> std::io::Result<Vec<InstalledModel>> {
        let r = http::get(&self.host, self.port, "/api/tags")?;
        Ok(parse_tags(&r.body))
    }

    /// Build a ModelSpec from the runtime's GGUF metadata.
    pub fn describe(&self, model: &InstalledModel) -> std::io::Result<Option<ModelSpec>> {
        let body = format!(r#"{{"model":"{}"}}"#, http::json_escape(&model.name));
        let r = http::post_json(&self.host, self.port, "/api/show", &body)?;
        if r.status != 200 {
            return Ok(None);
        }
        Ok(parse_show(&r.body, model))
    }

    /// Run one measured generation.
    ///
    /// `nonce` must differ between runs. Ollama caches KV for repeated prompt
    /// prefixes, so an identical prompt would report a near-zero prefill time
    /// and silently corrupt the calibration.
    ///
    /// The prompt is deliberately long (~1200 tokens). Prefill rate measured
    /// over a handful of tokens is dominated by fixed per-request overhead and
    /// is far too noisy to fit a coefficient from — which is exactly what
    /// `implied_prefill_scale` needs.
    pub fn generate(
        &self,
        model: &str,
        num_predict: u32,
        num_ctx: u32,
        nonce: u64,
    ) -> std::io::Result<RunStats> {
        let prompt = measurement_prompt(nonce);
        let body = format!(
            r#"{{"model":"{}","prompt":"{}","stream":false,"options":{{"num_predict":{},"num_ctx":{},"temperature":0}}}}"#,
            http::json_escape(model),
            http::json_escape(&prompt),
            num_predict,
            num_ctx
        );
        let r = http::post_json(&self.host, self.port, "/api/generate", &body)?;
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

/// Build a long, varied prompt for measuring prefill.
///
/// Three requirements:
///   * Long enough that prefill throughput is signal, not per-request overhead.
///   * The nonce must come *first*, so a repeated run shares no cached prefix.
///   * Varied text rather than a repeated phrase, so tokenisation stays
///     representative instead of collapsing into one repeated token.
pub fn measurement_prompt(nonce: u64) -> String {
    const CLAUSES: [&str; 8] = [
        "the measurement records throughput across successive layers",
        "each block of weights is read once per generated token",
        "bandwidth rather than arithmetic determines the observed rate",
        "cache residency shifts the balance between memory and storage",
        "quantisation changes how many bytes cross the bus per parameter",
        "attention state grows with every token retained in context",
        "routing selects a small subset of experts for each position",
        "prefill processes the whole prompt in a single weight pass",
    ];
    // ~1200 tokens at roughly 4.5 characters per token.
    let mut out = format!("Session {nonce}. Reference text follows. ");
    let mut i = nonce as usize;
    while out.len() < 5_400 {
        i = i.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        out.push_str(CLAUSES[(i >> 33) % CLAUSES.len()]);
        out.push_str(". ");
    }
    out.push_str("\nSummarise the text above in one short paragraph.");
    out
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

    /// Prefill measured over a short prompt is mostly per-request overhead.
    /// The prompt must be long, varied, and nonce-prefixed.
    #[test]
    fn measurement_prompt_is_long_varied_and_nonce_prefixed() {
        let a = measurement_prompt(1);
        let b = measurement_prompt(2);
        assert!(a.len() > 5_000, "prompt too short: {} chars", a.len());
        assert!(a.starts_with("Session 1."), "nonce must lead the prompt");
        assert_ne!(a, b, "different nonces must produce different prompts");
        // Prefixes must diverge immediately or the runtime reuses cached KV.
        assert_ne!(&a[..40], &b[..40]);
        // Varied clauses, not one phrase repeated.
        assert!(a.matches("bandwidth rather than").count() < 40, "too repetitive");
    }

    #[test]
    fn parses_ollama_host_forms() {
        let e = parse_host("http://192.168.1.5:9999").unwrap();
        assert_eq!((e.host.as_str(), e.port), ("192.168.1.5", 9999));
        let e = parse_host("myhost").unwrap();
        assert_eq!((e.host.as_str(), e.port), ("myhost", 11434));
        let e = parse_host(":11500").unwrap();
        assert_eq!((e.host.as_str(), e.port), ("127.0.0.1", 11500));
        assert!(parse_host("").is_none());

        // The commonest spelling of all, and the one that used to fail.
        let e = parse_host("localhost:11434").unwrap();
        assert_eq!((e.host.as_str(), e.port), ("localhost", 11434));
    }

    /// Splitting on the last colon mangles IPv6: `[::1]:11434` kept its
    /// brackets (which resolve to nothing) and bare `::1` became host ":"
    /// on port 1.
    #[test]
    fn parses_ipv6_hosts() {
        let e = parse_host("[::1]:11434").unwrap();
        assert_eq!((e.host.as_str(), e.port), ("::1", 11434));

        let e = parse_host("[::1]").unwrap();
        assert_eq!((e.host.as_str(), e.port), ("::1", 11434));

        let e = parse_host("http://[fe80::1]:9999").unwrap();
        assert_eq!((e.host.as_str(), e.port), ("fe80::1", 9999));

        // Unbracketed: ambiguous, so the whole value is the host.
        let e = parse_host("::1").unwrap();
        assert_eq!((e.host.as_str(), e.port), ("::1", 11434));
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
        InstalledModel {
            name: "qwen3:4b".into(),
            size_bytes: 2_600_000_000,
            quant: "Q4_K_M".into(),
        }
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
