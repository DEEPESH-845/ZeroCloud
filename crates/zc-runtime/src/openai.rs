//! The OpenAI-compatible surface, shared by everything that speaks it.
//!
//! vLLM, MLX, RamaLama and Docker Model Runner all serve `/v1/models` and
//! `/v1/chat/completions` on adjacent ports. That makes them cheap to detect
//! and impossible to calibrate: the chat response carries a `usage` block with
//! token counts and no timing at all, so the only way to get a rate would be to
//! time the request from outside — which measures HTTP and scheduling as much
//! as inference. See the crate docs for why that is disqualifying.
//!
//! So they are detected, named, and reported as unusable for calibration. The
//! user learns their runtime was seen rather than silently ignored, and the
//! dataset stays clean.

use crate::{Endpoint, InstalledModel, RunStats, Runtime};
use zc_model::json;

/// Parse an OpenAI `/v1/models` list.
///
/// llama.cpp extends each entry with a `meta` object (`n_params`, `n_embd`,
/// `n_vocab`, `size`); everyone else sends `id` alone. Both are handled here so
/// the two callers cannot drift.
pub fn parse_models(body: &str) -> Vec<InstalledModel> {
    let mut out = Vec::new();
    for obj in json::array_objects(body, "data") {
        let Some(id) = json::string(obj, "id") else {
            continue;
        };
        out.push(InstalledModel {
            // A quantisation is not part of the OpenAI schema. It is usually
            // spelled in the filename, and `from_gguf_label` reads a label out
            // of a longer string safely enough for the family classification.
            quant: quant_from_name(&id),
            name: id,
            size_bytes: json::number(obj, "size").unwrap_or(0.0) as u64,
            params: json::number(obj, "n_params").map(|v| v as u64),
            n_embd: json::number(obj, "n_embd").map(|v| v as u32),
            n_vocab: json::number(obj, "n_vocab").map(|v| v as u32),
        });
    }
    out
}

/// Pull a GGUF quantisation label out of a model name or path.
///
/// `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` -> `Q4_K_M`. Returns `"unknown"`
/// rather than guessing: the quant family sets dequantisation cost, a
/// first-order term in decode speed, so a wrong guess is a wrong prediction.
pub fn quant_from_name(name: &str) -> String {
    for token in name.split(['-', '_', '.', '/', '\\', ' ']).rev() {
        let u = token.to_uppercase();
        // The label is split by the same separators, so rebuild from the first
        // token that looks like the start of one.
        if u.starts_with('Q') && u.len() >= 2 && u.as_bytes()[1].is_ascii_digit() {
            return full_label(name, &u);
        }
        if u.starts_with("IQ") || u.starts_with("MXFP") || u == "F16" || u == "BF16" {
            return full_label(name, &u);
        }
    }
    "unknown".into()
}

/// Recover the whole `Q4_K_M`-style label around a leading token.
fn full_label(name: &str, head: &str) -> String {
    let upper = name.to_uppercase();
    let Some(start) = upper.find(head) else {
        return head.to_string();
    };
    upper[start..]
        .split(['-', '.', '/', '\\', ' '])
        .next()
        .unwrap_or(head)
        .to_string()
}

/// A runtime we can see and list but must never calibrate from.
pub struct OpenAiCompat {
    name: &'static str,
    ep: Endpoint,
}

impl OpenAiCompat {
    /// Identify whatever is answering `/v1/models` here.
    ///
    /// `owned_by` is what separates these: vLLM, MLX and Ollama's own
    /// compatibility layer all answer identically otherwise. `fallback` is the
    /// port's conventional occupant, used only when `owned_by` says nothing
    /// recognisable.
    pub fn at(ep: Endpoint, fallback: &'static str) -> Option<Self> {
        let body = ep.get("/v1/models").ok().filter(|r| r.status == 200)?.body;
        Some(OpenAiCompat {
            name: owner_tag(&body).unwrap_or(fallback),
            ep,
        })
    }
}

/// Map the `owned_by` field to a stable runtime tag.
fn owner_tag(body: &str) -> Option<&'static str> {
    let obj = json::array_objects(body, "data").into_iter().next()?;
    let owner = json::string(obj, "owned_by")?.to_lowercase();
    Some(match owner.as_str() {
        o if o.contains("vllm") => "vllm",
        o if o.contains("mlx") => "mlx",
        o if o.contains("llamacpp") || o.contains("llama.cpp") => "llamacpp",
        o if o.contains("ollama") => "ollama",
        _ => return None,
    })
}

/// Every OpenAI-compatible server listening on a port we know about.
pub fn detect_all(claimed: &[Endpoint]) -> Vec<OpenAiCompat> {
    // MLX shares 8080 with llama.cpp, which is why llama.cpp is probed first
    // and its endpoint arrives here already claimed.
    [
        (Endpoint::local(8000), "vllm"),
        (Endpoint::local(8080), "mlx"),
        (Endpoint::local(12434), "docker-model-runner"),
    ]
    .into_iter()
    .filter(|(ep, _)| !claimed.contains(ep))
    .filter_map(|(ep, fallback)| OpenAiCompat::at(ep, fallback))
    .collect()
}

impl Runtime for OpenAiCompat {
    fn name(&self) -> &'static str {
        self.name
    }

    fn endpoint(&self) -> &Endpoint {
        &self.ep
    }

    /// The whole reason this type exists separately.
    fn calibratable(&self) -> bool {
        false
    }

    fn list(&self) -> std::io::Result<Vec<InstalledModel>> {
        Ok(parse_models(&self.ep.get("/v1/models")?.body))
    }

    fn generate(&self, _: &str, _: u32, _: u32, _: u64) -> std::io::Result<RunStats> {
        Err(std::io::Error::other(format!(
            "{} does not report its own prefill/decode timings, so a measurement \
             taken here would include HTTP and scheduling overhead",
            self.name
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// llama.cpp's extended entry carries the facts that let a name match be
    /// verified rather than trusted.
    #[test]
    fn parses_llama_cpp_extended_model_entries() {
        let body = r#"{"object":"list","data":[{"id":"../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "object":"model","created":1735142223,"owned_by":"llamacpp",
            "meta":{"vocab_type":2,"n_vocab":128256,"n_ctx_train":131072,
            "n_embd":4096,"n_params":8030261312,"size":4912898304}}]}"#;
        let v = parse_models(body);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].size_bytes, 4_912_898_304);
        assert_eq!(v[0].params, Some(8_030_261_312));
        assert_eq!(v[0].n_embd, Some(4096));
        assert_eq!(v[0].quant, "Q4_K_M");
    }

    /// A plain OpenAI server says only `id`. Nothing may be invented from that
    /// — every optional fact stays `None` so no catalog match can be faked.
    #[test]
    fn a_bare_openai_entry_invents_nothing() {
        let v = parse_models(r#"{"data":[{"id":"granite-3.0-2b-instruct","owned_by":"vllm"}]}"#);
        assert_eq!(v[0].size_bytes, 0);
        assert_eq!(v[0].params, None);
        assert_eq!(v[0].quant, "unknown");
    }

    #[test]
    fn quant_labels_are_recovered_from_file_names() {
        assert_eq!(quant_from_name("Meta-Llama-3.1-8B-Q4_K_M.gguf"), "Q4_K_M");
        assert_eq!(quant_from_name("/m/qwen3-30b-a3b-IQ3_XXS.gguf"), "IQ3_XXS");
        assert_eq!(quant_from_name("gpt-oss-20b-MXFP4.gguf"), "MXFP4");
        // No label anywhere: say so rather than defaulting to a family whose
        // dequantisation cost would be wrong.
        assert_eq!(quant_from_name("granite-3.0-2b-instruct"), "unknown");
    }

    #[test]
    fn owned_by_disambiguates_servers_on_adjacent_ports() {
        assert_eq!(owner_tag(r#"{"data":[{"id":"m","owned_by":"vllm"}]}"#), Some("vllm"));
        assert_eq!(owner_tag(r#"{"data":[{"id":"m","owned_by":"mlx-lm"}]}"#), Some("mlx"));
        assert_eq!(owner_tag(r#"{"data":[{"id":"m","owned_by":"library"}]}"#), None);
    }
}
