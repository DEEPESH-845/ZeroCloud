//! The model catalog, loaded from `data/models/*.json`.
//!
//! Kept as data rather than Rust for one reason: it is the community
//! contribution surface. Adding a model must be a one-file PR that needs no
//! Rust knowledge and cannot conflict with anyone else's addition.
//!
//! Files are embedded at build time (see `build.rs`) so the binary works
//! offline, and a directory on disk can override them so contributors can test
//! a new entry without recompiling.

use crate::json;
use crate::spec::*;

include!(concat!(env!("OUT_DIR"), "/catalog_embedded.rs"));

fn parse_attention(obj: &str) -> Option<Attention> {
    let n = |k: &str| json::number(obj, k).map(|v| v as u32);
    match json::string(obj, "kind")?.as_str() {
        "gqa" => Some(Attention::Gqa {
            n_kv_heads: n("n_kv_heads")?,
            head_dim: n("head_dim")?,
        }),
        "mla" => Some(Attention::Mla {
            kv_lora_rank: n("kv_lora_rank")?,
            rope_head_dim: n("rope_head_dim")?,
        }),
        "swa" => Some(Attention::Swa {
            n_kv_heads: n("n_kv_heads")?,
            head_dim: n("head_dim")?,
            window: n("window")?,
            global_every: n("global_every").unwrap_or(6),
        }),
        "hybrid" => Some(Attention::Hybrid {
            attn_layers: n("attn_layers")?,
            n_kv_heads: n("n_kv_heads")?,
            head_dim: n("head_dim")?,
            ssm_state_bytes: json::number(obj, "ssm_state_bytes")? as u64,
        }),
        // An unknown kind is a data error. Returning None drops the entry with
        // a warning rather than silently predicting with the wrong KV formula,
        // which would be worse than showing nothing.
        _ => None,
    }
}

/// Parse one catalog file. Returns `None` on any missing required field.
pub fn parse_model(src: &str) -> Option<ModelSpec> {
    let num = |k: &str| json::number(src, k);
    let spec = ModelSpec {
        id: json::string(src, "id")?,
        n_layers: num("n_layers")? as u32,
        n_embd: num("n_embd")? as u32,
        n_vocab: num("n_vocab")? as u32,
        params: num("params")? as u64,
        attention: parse_attention(json::object_at(src, "attention")?)?,
        moe: json::object_at(src, "moe").and_then(|o| {
            Some(Moe {
                n_expert: json::number(o, "n_expert")? as u32,
                n_active: json::number(o, "n_active")? as u32,
                expert_params: json::number(o, "expert_params")? as u64,
                shared_params: json::number(o, "shared_params")? as u64,
            })
        }),
        quants: json::array_objects(src, "quants")
            .iter()
            .filter_map(|q| {
                Some(Quant {
                    name: json::string(q, "name")?,
                    bytes: json::number(q, "bytes")? as u64,
                    family: QuantFamily::from_data_tag(&json::string(q, "family").unwrap_or_default()),
                })
            })
            .collect(),
        };
    // A model with no parseable quantisation cannot be predicted for.
    (!spec.quants.is_empty()).then_some(spec)
}

/// Every model in the embedded catalog, plus any override directory.
///
/// Precedence: a file on disk with the same `id` replaces the embedded entry,
/// so a contributor can iterate on `data/models/` without rebuilding.
pub fn load() -> Vec<ModelSpec> {
    let mut out: Vec<ModelSpec> = EMBEDDED.iter().filter_map(|s| parse_model(s)).collect();

    let dir = std::env::var("ZC_DATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("data/models"));
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for path in entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
        {
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            match parse_model(&text) {
                Some(m) => match out.iter().position(|e| e.id == m.id) {
                    Some(i) => out[i] = m,
                    None => out.push(m),
                },
                None => eprintln!("warning: skipping unparseable catalog file {}", path.display()),
            }
        }
    }

    out.sort_by_key(|m| m.params);
    out
}

/// Kept for callers that want the compiled-in set only, ignoring disk.
pub fn builtin() -> Vec<ModelSpec> {
    let mut out: Vec<ModelSpec> = EMBEDDED.iter().filter_map(|s| parse_model(s)).collect();
    out.sort_by_key(|m| m.params);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KvPrecision;

    /// Every shipped file must parse. A catalog entry that silently fails to
    /// load is a model the user is told they cannot run.
    #[test]
    fn every_embedded_file_parses() {
        assert!(!EMBEDDED.is_empty(), "build.rs embedded nothing");
        for (i, src) in EMBEDDED.iter().enumerate() {
            assert!(parse_model(src).is_some(), "embedded file {i} failed to parse");
        }
        assert_eq!(builtin().len(), EMBEDDED.len());
    }

    fn get(id: &str) -> ModelSpec {
        builtin()
            .into_iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| panic!("{id} missing from catalog"))
    }

    /// Validates the *data*, not just the parser: these are hand-computed and
    /// any of them changing means a catalog file has a wrong architecture
    /// field, which would silently corrupt every prediction for that model.
    #[test]
    fn catalog_data_matches_hand_computed_kv() {
        // 2 * 32 layers * 8 kv heads * 128 dim * 2 bytes
        assert_eq!(get("llama-3.1-8b").kv_bytes(1, KvPrecision::F16), 131_072);
        // 2 * 80 * 8 * 128 * 2 -> exactly 10 GiB at 32K
        assert_eq!(get("llama-3.3-70b").kv_bytes(1, KvPrecision::F16), 327_680);
        assert_eq!(
            get("llama-3.3-70b").kv_bytes(32768, KvPrecision::F16),
            10 << 30
        );
        // 2 * 36 * 8 * 128 * 2
        assert_eq!(get("qwen3-4b").kv_bytes(1, KvPrecision::F16), 147_456);
        // MLA: 61 layers * (512 + 64) * 2, no factor of two for K and V
        assert_eq!(get("deepseek-v3").kv_bytes(1, KvPrecision::F16), 70_272);
    }

    /// The MoE geometry in the catalog must reproduce the advertised active
    /// parameter count. Qwen3-30B-A3B is named for its 3B active params.
    #[test]
    fn moe_geometry_reproduces_advertised_active_params() {
        let m = get("qwen3-30b-a3b");
        let active_b = m.active_params() as f64 / 1e9;
        assert!((3.0..3.7).contains(&active_b), "active {active_b}B, expected ~3.3B");

        let ds = get("deepseek-v3");
        let active_b = ds.active_params() as f64 / 1e9;
        assert!((30.0..40.0).contains(&active_b), "active {active_b}B, expected ~37B");
    }

    #[test]
    fn sliding_window_model_is_parsed_as_swa() {
        assert!(matches!(
            get("gemma-3-12b").attention,
            Attention::Swa { window: 1024, global_every: 6, .. }
        ));
    }

    #[test]
    fn quant_families_round_trip_from_data() {
        let m = get("qwen3-30b-a3b");
        assert_eq!(m.quants[0].family, QuantFamily::KQuant);
        assert_eq!(m.quants[1].family, QuantFamily::IQuant);
    }

    /// Malformed or incomplete entries must be dropped, never half-loaded.
    #[test]
    fn invalid_entries_are_rejected() {
        assert!(parse_model("{}").is_none());
        // Missing head_dim.
        assert!(parse_model(
            r#"{"id":"x","n_layers":1,"n_embd":1,"n_vocab":1,"params":1,
                "attention":{"kind":"gqa","n_kv_heads":8},
                "quants":[{"name":"Q4","bytes":1,"family":"k_quant"}]}"#
        )
        .is_none());
        // Unknown attention kind: we must not guess a KV formula.
        assert!(parse_model(
            r#"{"id":"x","n_layers":1,"n_embd":1,"n_vocab":1,"params":1,
                "attention":{"kind":"future-thing"},
                "quants":[{"name":"Q4","bytes":1,"family":"k_quant"}]}"#
        )
        .is_none());
        // No quantisations means nothing to predict.
        assert!(parse_model(
            r#"{"id":"x","n_layers":1,"n_embd":1,"n_vocab":1,"params":1,
                "attention":{"kind":"gqa","n_kv_heads":8,"head_dim":128},
                "quants":[]}"#
        )
        .is_none());
    }

    /// Catalog is sorted small-to-large so the report leads with what is most
    /// likely to run on a constrained machine.
    #[test]
    fn catalog_is_sorted_by_size() {
        let v = builtin();
        assert!(v.windows(2).all(|w| w[0].params <= w[1].params));
        assert_eq!(v[0].id, "qwen3-4b");
    }
}
