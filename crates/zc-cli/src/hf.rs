//! `zc check <hf-repo-id>` — will this model, the one nobody has catalogued,
//! fit on this machine?
//!
//! The catalog covers 26 models. The question people actually ask is about the
//! one they just saw posted, so this reads the repository's own metadata and
//! answers from it.
//!
//! **What it does not answer is decode speed, and that is deliberate.** Speed
//! needs the byte count of the *quantised* file, which lives in a separate GGUF
//! repository the user has not named. The alternatives were to demand a second
//! repo argument, or to estimate bytes as `params x bits-per-weight` — and the
//! second would put an unmeasured number inside a prediction for the first time
//! in this codebase. So speed prints a dash, exactly as TTFT does when nothing
//! has measured it.
//!
//! Memory is a different matter: `safetensors.parameters` reports the parameter
//! count *per dtype*, so the published weight size is arithmetic over numbers
//! the repository states, not a guess.
//!
//! # Network
//!
//! This is the only subcommand that opens an outbound connection, and only when
//! given a repository id. It prints each URL before fetching it. Nothing about
//! the machine is sent — the request carries a repo id and nothing else.

use zc_model::json;
use zc_model::spec::{Attention, ModelSpec};

const API: &str = "https://huggingface.co/api/models";
const RAW: &str = "https://huggingface.co";
/// Metadata is kilobytes. A repo that cannot answer in this long is down.
const TIMEOUT_SECS: u32 = 20;

/// Does this look like a repository id rather than a flag or a model name?
///
/// One slash, no spaces, and not a filesystem path. Deliberately strict: a
/// wrong guess here turns a typo into a network request.
pub fn looks_like_repo_id(s: &str) -> bool {
    let parts: Vec<&str> = s.split('/').collect();
    parts.len() == 2
        && parts.iter().all(|p| {
            !p.is_empty()
                && p.chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        })
        && !s.starts_with('.')
}

/// Fetch a URL as text.
///
/// Shells out to `curl` rather than linking a TLS stack. `zc-probe` already
/// shells out to `nvidia-smi`, `lspci` and `powershell` for the same reason:
/// rustls and its dependencies would roughly triple a tree that is currently
/// libc plus crossterm, to make two requests that most users never make.
fn fetch(url: &str) -> Result<String, String> {
    eprintln!("  fetching {url}");
    let out = std::process::Command::new("curl")
        .args([
            "-fsSL",
            "--max-time",
            &TIMEOUT_SECS.to_string(),
            "-H",
            "Accept: application/json",
            url,
        ])
        .output()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => {
                "curl is not installed, and it is how zc makes this one request".to_string()
            }
            _ => format!("could not run curl: {e}"),
        })?;
    if !out.status.success() {
        let why = String::from_utf8_lossy(&out.stderr);
        let why = why.trim().to_string();
        // Hugging Face answers 401 for gated, private *and* nonexistent repos
        // alike -- deliberately, so a 404 cannot be used to probe for private
        // repositories. Naming only one of the three would be a guess, and the
        // wrong guess sends people hunting a bug that is not there.
        if why.contains("401") || why.contains("403") {
            return Err(format!(
                "{url}\n  returned 401. Hugging Face answers the same for a gated repo, a\n  private one, and one that does not exist. If it is gated (Llama and\n  Gemma are), accept its licence on huggingface.co -- zc holds no token\n  and asks for none."
            ));
        }
        if why.contains("404") {
            return Err(format!("no such repo, or it has no config.json:\n  {url}"));
        }
        return Err(if why.is_empty() {
            format!("{url} could not be read")
        } else {
            why
        });
    }
    String::from_utf8(out.stdout).map_err(|_| "response was not UTF-8".to_string())
}

/// Bytes per element for the dtypes safetensors reports.
///
/// Anything unrecognised makes the total unknown rather than approximate: a
/// weight size that silently omits a tensor is worse than no weight size.
fn dtype_bytes(tag: &str) -> Option<u64> {
    Some(match tag {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "F8_E4M3" | "F8_E5M2" | "I8" | "U8" | "BOOL" => 1,
        _ => return None,
    })
}

pub struct Fetched {
    pub spec: ModelSpec,
    /// Published weight size in bytes, summed per dtype.
    pub weight_bytes: u64,
    /// The dtypes the weights are stored in, largest share first.
    pub dtypes: String,
}

/// Parse the two documents into something the memory model can use.
///
/// Split from the fetching so the mapping is testable without a network.
pub fn parse(id: &str, api: &str, config: &str) -> Result<Fetched, String> {
    let cfg_num = |k: &str| json::number(config, k).map(|v| v as u32);
    let n_layers = cfg_num("num_hidden_layers")
        .ok_or("config.json has no num_hidden_layers — not a transformer this can read")?;
    let n_embd = cfg_num("hidden_size").ok_or("config.json has no hidden_size")?;
    let n_vocab = cfg_num("vocab_size").ok_or("config.json has no vocab_size")?;
    let n_heads = cfg_num("num_attention_heads").unwrap_or(0);
    let n_kv_heads = cfg_num("num_key_value_heads").unwrap_or(n_heads);
    if n_kv_heads == 0 {
        return Err("config.json states no attention head counts, so KV size is not derivable"
            .to_string());
    }
    let head_dim = cfg_num("head_dim")
        .or_else(|| n_embd.checked_div(n_heads))
        .filter(|d| *d > 0)
        .ok_or("config.json states no head_dim and none can be derived")?;

    // Sliding window changes the KV maths by more than an order of magnitude,
    // and `use_sliding_window: false` alongside a stated window is common —
    // believing the number there would understate memory badly.
    let window = cfg_num("sliding_window")
        .filter(|w| *w > 0)
        .filter(|_| json::boolean(config, "use_sliding_window").unwrap_or(true));

    let attention = match window {
        Some(window) => Attention::Swa {
            n_kv_heads,
            head_dim,
            window,
            global_every: cfg_num("sliding_window_pattern").unwrap_or(1).max(1),
        },
        None => Attention::Gqa {
            n_kv_heads,
            head_dim,
        },
    };

    // Parameter counts per dtype. This is what makes the weight size arithmetic
    // rather than a guess.
    let params_obj = json::object_at(api, "parameters")
        .ok_or("the API reports no safetensors metadata for this repo")?;
    let mut weight_bytes: u64 = 0;
    let mut total_params: u64 = 0;
    let mut seen: Vec<(String, u64)> = Vec::new();
    for tag in [
        "F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2", "I64", "I32", "I16", "I8", "U8", "BOOL",
    ] {
        if let Some(n) = json::number(params_obj, tag) {
            let n = n as u64;
            if n == 0 {
                continue;
            }
            let per = dtype_bytes(tag).ok_or(format!("unrecognised dtype {tag}"))?;
            weight_bytes += n * per;
            total_params += n;
            seen.push((tag.to_string(), n));
        }
    }
    if total_params == 0 {
        return Err("the API reports no parameter counts for this repo".to_string());
    }
    seen.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    let dtypes = seen
        .iter()
        .map(|(t, _)| t.as_str())
        .collect::<Vec<_>>()
        .join(" + ");

    let n_ctx_train = cfg_num("max_position_embeddings").filter(|v| *v > 0);

    Ok(Fetched {
        spec: ModelSpec {
            id: id.to_string(),
            n_layers,
            n_embd,
            n_vocab,
            params: total_params,
            attention,
            // MoE routing is not derivable from config.json in a portable way,
            // and it changes only decode speed, which this path does not report.
            moe: None,
            n_ctx_train,
            quants: Vec::new(),
        },
        weight_bytes,
        dtypes,
    })
}

pub fn fetch_model(id: &str) -> Result<Fetched, String> {
    let api = fetch(&format!("{API}/{id}"))?;
    let config = fetch(&format!("{RAW}/{id}/raw/main/config.json"))?;
    parse(id, &api, &config)
}

/// What this machine can do with the model, on memory grounds alone.
///
/// Deliberately its own block rather than a row in the model table: a table row
/// promises a decode speed, and the honest answer here is that nobody has
/// measured the quantised file this machine would actually load.
pub fn render(f: &Fetched, budget: u64, kv: zc_model::KvPrecision, ubatch: u32) -> String {
    use std::fmt::Write;
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let mut o = String::new();
    let s = &f.spec;

    let _ = writeln!(o, "\n== {} ==  (published weights, not a quantisation)\n", s.id);

    let attn = match s.attention {
        Attention::Gqa {
            n_kv_heads,
            head_dim,
        } => format!("GQA {n_kv_heads}x{head_dim}"),
        Attention::Swa {
            n_kv_heads,
            head_dim,
            window,
            global_every,
        } => format!("SWA {n_kv_heads}x{head_dim} window {window}, global every {global_every}"),
        Attention::Mla { .. } => "MLA".to_string(),
        // Not reachable from this path -- `parse` only ever builds Gqa or Swa
        // -- but the enum is shared with the catalog, which has both.
        _ => format!("{:?}", s.attention),
    };
    let _ = writeln!(
        o,
        "  geometry     {}L  n_embd {}  vocab {}  {attn}",
        s.n_layers, s.n_embd, s.n_vocab
    );
    let _ = writeln!(
        o,
        "  parameters   {:.2}B  from the repo's safetensors metadata",
        s.params as f64 / 1e9
    );
    let _ = writeln!(
        o,
        "  weights      {:.2} GiB  as published in {}",
        gib(f.weight_bytes),
        f.dtypes
    );

    let bufs = s.compute_buffer_bytes(ubatch);
    let after = budget.saturating_sub(bufs);
    let for_kv = after.saturating_sub(f.weight_bytes);
    let ctx = s
        .max_context_in(for_kv, kv)
        .min(s.n_ctx_train.unwrap_or(u32::MAX));
    let per_token = s.kv_bytes_per_token(ctx.max(1), kv);

    let _ = writeln!(
        o,
        "  budget       {:.2} GiB  measured on this machine, idle",
        gib(budget)
    );
    let _ = writeln!(
        o,
        "  KV           {:.2} MiB/token at {}",
        per_token as f64 / (1u64 << 20) as f64,
        kv.tag().to_uppercase().to_uppercase()
    );

    if f.weight_bytes + bufs >= budget {
        let _ = writeln!(
            o,
            "\n  WON'T FIT    the published weights alone are {:.2} GiB over budget",
            gib(f.weight_bytes + bufs - budget)
        );
        let _ = writeln!(
            o,
            "               a quantisation may still fit -- Q4_K_M is roughly a"
        );
        let _ = writeln!(o, "               quarter of the size of BF16 weights");
    } else if ctx < 512 {
        let _ = writeln!(
            o,
            "\n  WON'T FIT    weights fit, but no usable context fits beside them"
        );
    } else {
        let _ = writeln!(
            o,
            "\n  FITS         up to {} context at {}",
            if ctx >= 1024 {
                format!("{}K", ctx / 1024)
            } else {
                ctx.to_string()
            },
            kv.tag().to_uppercase()
        );
        if s.n_ctx_train.is_some_and(|t| ctx >= t) {
            let _ = writeln!(
                o,
                "               capped by what the model was trained for, not by memory"
            );
        }
    }

    // The whole reason this path exists, stated where it cannot be missed.
    let _ = writeln!(
        o,
        "\n  decode       -   speed needs the byte count of the quantised file you"
    );
    let _ = writeln!(
        o,
        "                   would actually load, which lives in a separate GGUF"
    );
    let _ = writeln!(
        o,
        "                   repo. Memory above is arithmetic over numbers the"
    );
    let _ = writeln!(
        o,
        "                   repo states; a speed here would not be. Run"
    );
    let _ = writeln!(
        o,
        "                   `zc verify` once you have pulled it to measure both."
    );
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    const API: &str = r#"{"id":"x","safetensors":{"parameters":{"BF16":8030261248},"total":8030261248}}"#;
    const CFG: &str = r#"{"num_hidden_layers":32,"hidden_size":4096,"vocab_size":128256,
        "num_attention_heads":32,"num_key_value_heads":8,"max_position_embeddings":131072}"#;

    #[test]
    fn a_repo_id_is_recognised_but_a_path_is_not() {
        assert!(looks_like_repo_id("meta-llama/Llama-3.1-8B"));
        assert!(looks_like_repo_id("Qwen/Qwen3-1.7B"));
        assert!(!looks_like_repo_id("qwen3:1.7b"));
        assert!(!looks_like_repo_id("./data/models"));
        assert!(!looks_like_repo_id("a/b/c"));
        assert!(!looks_like_repo_id("/etc/passwd"));
        assert!(!looks_like_repo_id("--json"));
        assert!(!looks_like_repo_id("has space/x"));
    }

    /// The weight size is arithmetic over what the repo states, per dtype —
    /// never params x an assumed width.
    #[test]
    fn weight_bytes_come_from_the_dtype_breakdown() {
        let f = parse("meta-llama/Llama-3.1-8B", API, CFG).unwrap();
        assert_eq!(f.spec.params, 8_030_261_248);
        assert_eq!(f.weight_bytes, 8_030_261_248 * 2, "BF16 is two bytes");
        assert_eq!(f.dtypes, "BF16");
        assert_eq!(f.spec.n_layers, 32);
        assert_eq!(f.spec.n_ctx_train, Some(131072));
    }

    #[test]
    fn mixed_dtypes_are_summed_separately() {
        let api = r#"{"safetensors":{"parameters":{"BF16":1000,"F32":10}}}"#;
        let f = parse("a/b", api, CFG).unwrap();
        assert_eq!(f.weight_bytes, 1000 * 2 + 10 * 4);
        assert_eq!(f.spec.params, 1010);
        assert_eq!(f.dtypes, "BF16 + F32", "largest share first");
    }

    /// A missing count is a refusal, not a zero: a weight size that silently
    /// omits tensors is worse than no weight size.
    #[test]
    fn missing_metadata_is_refused_rather_than_guessed() {
        assert!(parse("a/b", r#"{"safetensors":{}}"#, CFG).is_err());
        assert!(parse("a/b", API, r#"{"hidden_size":4096}"#).is_err());
        let no_heads = r#"{"num_hidden_layers":32,"hidden_size":4096,"vocab_size":128256}"#;
        assert!(parse("a/b", API, no_heads).is_err());
    }

    /// `sliding_window` alongside `use_sliding_window: false` is common, and
    /// believing it would understate KV memory by an order of magnitude.
    #[test]
    fn a_disabled_sliding_window_is_ignored() {
        let on = format!("{}{}", &CFG[..CFG.len() - 1], r#","sliding_window":4096}"#);
        assert!(matches!(
            parse("a/b", API, &on).unwrap().spec.attention,
            Attention::Swa { window: 4096, .. }
        ));
        let off = format!(
            "{}{}",
            &CFG[..CFG.len() - 1],
            r#","sliding_window":4096,"use_sliding_window":false}"#
        );
        assert!(matches!(
            parse("a/b", API, &off).unwrap().spec.attention,
            Attention::Gqa { .. }
        ));
    }

    /// head_dim is derived from hidden_size / heads when not stated, which is
    /// how most configs express it.
    #[test]
    fn head_dim_is_derived_when_absent() {
        let f = parse("a/b", API, CFG).unwrap();
        assert!(matches!(
            f.spec.attention,
            Attention::Gqa {
                n_kv_heads: 8,
                head_dim: 128
            }
        ));
    }
}
