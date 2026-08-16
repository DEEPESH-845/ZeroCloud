//! Talking to installed inference runtimes, and turning what they report into
//! calibration data.
//!
//! This crate is what converts every constant in `zc-model` from a guess into
//! a measurement. Nothing else in the project can do that.
//!
//! # The rule every runtime here is judged by
//!
//! **A runtime may only write calibration records if it reports its own
//! prefill/decode split.** Timing a generation from outside includes HTTP
//! framing, model load, and process scheduling; folding that into `implied_eta`
//! would corrupt the one dataset this project cannot rebuild. A runtime that
//! cannot supply the split is still worth detecting and listing — the user
//! should know it is there — but [`Runtime::calibratable`] returns false and
//! `zc verify` refuses to record from it.

pub mod calibrate;
pub mod http;
pub mod llamacpp;
pub mod lmstudio;
pub mod ollama;
pub mod openai;

use zc_model::spec::ModelSpec;

/// Where a runtime lives.
///
/// Carried explicitly rather than hardcoded for two reasons: users genuinely
/// do run these on another host or port (Ollama itself reads `OLLAMA_HOST`),
/// and a hardcoded endpoint makes the happy path impossible to test without
/// installing a runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoint {
    pub host: String,
    pub port: u16,
}

impl Endpoint {
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            host: host.to_string(),
            port,
        }
    }

    /// Loopback on `port`. Every runtime here defaults to listening locally.
    pub fn local(port: u16) -> Self {
        Self::new("127.0.0.1", port)
    }

    /// Read `var` from the environment, falling back to loopback on
    /// `default_port`.
    pub fn from_env(var: &str, default_port: u16) -> Self {
        std::env::var(var)
            .ok()
            .and_then(|v| parse_host(&v, default_port))
            .unwrap_or_else(|| Self::local(default_port))
    }

    pub fn get(&self, path: &str) -> std::io::Result<http::Response> {
        http::get(&self.host, self.port, path)
    }

    pub fn post(&self, path: &str, body: &str) -> std::io::Result<http::Response> {
        http::post_json(&self.host, self.port, path, body)
    }

    /// Whether `path` answers 200. The cheapest possible liveness probe.
    pub fn responds(&self, path: &str) -> bool {
        self.get(path).is_ok_and(|r| r.status == 200)
    }
}

impl std::fmt::Display for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.host, self.port)
    }
}

/// Parse a `*_HOST` value: `host`, `host:port`, or `http://host:port`.
///
/// Split from [`Endpoint::from_env`] so it can be tested without mutating
/// process environment, which is global state and races with parallel tests.
pub fn parse_host(raw: &str, default_port: u16) -> Option<Endpoint> {
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
                None => default_port,
            },
        });
    }
    // A bare IPv6 address has several colons and no port; splitting on the last
    // one turned `::1` into host ":" port 1.
    if raw.matches(':').count() > 1 {
        return Some(Endpoint {
            host: raw.into(),
            port: default_port,
        });
    }
    Some(match raw.rsplit_once(':') {
        Some((h, p)) => Endpoint {
            host: if h.is_empty() {
                "127.0.0.1".into()
            } else {
                h.into()
            },
            port: p.parse().ok()?,
        },
        None => Endpoint {
            host: raw.into(),
            port: default_port,
        },
    })
}

/// A model a runtime says it has.
///
/// The optional fields are facts the runtime volunteered. They are `None` when
/// it said nothing, never a default — a guessed parameter count would silently
/// match the wrong catalog entry, which is worse than failing to match at all.
#[derive(Debug, Clone, Default)]
pub struct InstalledModel {
    pub name: String,
    /// On-disk size in bytes — the model's quantised weight size. 0 when the
    /// runtime does not report it.
    pub size_bytes: u64,
    pub quant: String,
    pub params: Option<u64>,
    pub n_embd: Option<u32>,
    pub n_vocab: Option<u32>,
}

impl InstalledModel {
    pub fn new(name: &str, size_bytes: u64, quant: &str) -> Self {
        Self {
            name: name.to_string(),
            size_bytes,
            quant: quant.to_string(),
            ..Default::default()
        }
    }
}

/// What the runtime measured, as opposed to what we predicted.
#[derive(Debug, Clone, Default)]
pub struct RunStats {
    pub model: String,
    pub prompt_tokens: u32,
    pub eval_tokens: u32,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    pub load_s: f64,
    /// Context the run actually used, when the runtime reports it.
    ///
    /// `None` means it honoured whatever we asked for. llama.cpp and LM Studio
    /// fix context at server or load time and ignore a per-request setting, so
    /// recording our request rather than their reality would put a wrong number
    /// into the shared dataset.
    pub n_ctx: Option<u32>,
}

/// One local inference runtime.
///
/// Deliberately small: detect, list, describe, generate. Downloading and
/// running models is a package manager's job, not an oracle's.
pub trait Runtime {
    /// Stable tag, recorded in every calibration record.
    fn name(&self) -> &'static str;

    fn endpoint(&self) -> &Endpoint;

    /// Whether this runtime reports its own prefill/decode split. See the
    /// module docs — this gates whether a record may be written at all.
    fn calibratable(&self) -> bool {
        true
    }

    fn list(&self) -> std::io::Result<Vec<InstalledModel>>;

    /// Full model geometry, when the runtime exposes it.
    ///
    /// Only Ollama does today: it serves the GGUF metadata verbatim. The
    /// others report a name and little else, so callers fall back to
    /// [`catalog_match`]. Returning `None` rather than a partial guess is the
    /// point — attention geometry cannot be inferred from a filename.
    fn describe(&self, _model: &InstalledModel) -> std::io::Result<Option<ModelSpec>> {
        Ok(None)
    }

    /// Run one measured generation.
    ///
    /// `nonce` must differ between runs: every runtime here caches KV for a
    /// repeated prompt prefix, and a cache hit reports a near-zero prefill time
    /// that would silently corrupt the calibration.
    fn generate(
        &self,
        model: &str,
        num_predict: u32,
        num_ctx: u32,
        nonce: u64,
    ) -> std::io::Result<RunStats>;
}

/// Every runtime currently listening, in the order they are preferred.
///
/// The order is load-bearing, not cosmetic:
///
/// * Ollama first — it is the only one that reports full model metadata, so it
///   can calibrate any installed model rather than only catalogued ones.
/// * llama.cpp is identified **positively** through `/props` before anything
///   else is tried on port 8080, because MLX's server defaults to the same port
///   and answering `/v1/models` there does not make it llama.cpp.
/// * OpenAI-compatible servers last: they are detected so the user knows they
///   were seen, but none of them reports a prefill/decode split.
pub fn detect() -> Vec<Box<dyn Runtime>> {
    let mut found: Vec<Box<dyn Runtime>> = Vec::new();
    let mut claimed: Vec<Endpoint> = Vec::new();

    if let Some(rt) = ollama::Ollama::detect() {
        claimed.push(rt.endpoint().clone());
        found.push(Box::new(rt));
    }
    if let Some(rt) = lmstudio::LmStudio::detect() {
        claimed.push(rt.endpoint().clone());
        found.push(Box::new(rt));
    }
    if let Some(rt) = llamacpp::LlamaCpp::detect() {
        claimed.push(rt.endpoint().clone());
        found.push(Box::new(rt));
    }
    for rt in openai::detect_all(&claimed) {
        found.push(Box::new(rt));
    }
    found
}

/// Find the catalog entry a runtime-reported model corresponds to.
///
/// llama.cpp and LM Studio name a model but do not describe its architecture,
/// and KV geometry cannot be inferred from a name. The catalog has the
/// geometry; this is the join.
///
/// The match is deliberately hard to pass. A name match alone would happily
/// pair `llama-3.1-8b` with a 70B file, so every fact the runtime *did* report
/// is checked against the entry, and any disagreement rejects the match. A
/// missing model is recoverable; a confidently wrong one is not.
pub fn catalog_match(model: &InstalledModel) -> Option<ModelSpec> {
    let want = squash(&model.name);
    let mut spec = zc_model::catalog::load()
        .into_iter()
        .filter(|s| {
            let id = squash(&s.id);
            want.contains(&id) || id.contains(&want)
        })
        // Longest id wins: `llama-3.1-8b` and `llama-3.1-8b-instruct` both
        // match a file named for the latter, and the more specific entry is
        // the right one.
        .max_by_key(|s| s.id.len())?;

    // Cross-check whatever the runtime volunteered. Within 2% absorbs the
    // difference between an advertised parameter count and a counted one.
    if let Some(p) = model.params
        && (p as f64 - spec.params as f64).abs() / spec.params as f64 > 0.02
    {
        return None;
    }
    if model.n_embd.is_some_and(|v| v != spec.n_embd) {
        return None;
    }
    if model.n_vocab.is_some_and(|v| v != spec.n_vocab) {
        return None;
    }

    // Keep only the quantisation actually installed, and prefer the runtime's
    // own byte count: it is the real file, where the catalog carries a
    // representative one.
    let label = model.quant.to_uppercase();
    let mut quants: Vec<_> = spec
        .quants
        .iter()
        .filter(|q| q.name.to_uppercase() == label)
        .cloned()
        .collect();
    if quants.is_empty() {
        // The catalog does not list this quantisation. We still know its family
        // and, from the runtime, its size — that is everything `predict` needs.
        //
        // Unless the label itself is missing: quant family sets dequantisation
        // cost, and the spread between legacy (0.95) and i-quant (0.72) is far
        // too wide to default through. No label, no prediction.
        if model.size_bytes == 0 || model.quant.eq_ignore_ascii_case("unknown") {
            return None;
        }
        quants.push(zc_model::spec::Quant {
            name: model.quant.clone(),
            bytes: model.size_bytes,
            family: zc_model::QuantFamily::from_gguf_label(&model.quant),
        });
    } else if model.size_bytes > 0 {
        quants[0].bytes = model.size_bytes;
    }
    spec.quants = quants;
    Some(spec)
}

/// Lowercase alphanumerics only, so `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
/// and `llama-3.1-8b` can be compared.
fn squash(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_host_forms_against_a_supplied_default_port() {
        let e = parse_host("http://192.168.1.5:9999", 11434).unwrap();
        assert_eq!((e.host.as_str(), e.port), ("192.168.1.5", 9999));
        // The default follows the runtime, not a constant: LM Studio is 1234.
        let e = parse_host("myhost", 1234).unwrap();
        assert_eq!((e.host.as_str(), e.port), ("myhost", 1234));
        let e = parse_host(":11500", 11434).unwrap();
        assert_eq!((e.host.as_str(), e.port), ("127.0.0.1", 11500));
        assert!(parse_host("", 8080).is_none());

        // The commonest spelling of all, and the one that used to fail.
        let e = parse_host("localhost:11434", 11434).unwrap();
        assert_eq!((e.host.as_str(), e.port), ("localhost", 11434));
    }

    /// Splitting on the last colon mangles IPv6: `[::1]:11434` kept its
    /// brackets (which resolve to nothing) and bare `::1` became host ":"
    /// on port 1.
    #[test]
    fn parses_ipv6_hosts() {
        let e = parse_host("[::1]:11434", 11434).unwrap();
        assert_eq!((e.host.as_str(), e.port), ("::1", 11434));
        let e = parse_host("[::1]", 8080).unwrap();
        assert_eq!((e.host.as_str(), e.port), ("::1", 8080));
        let e = parse_host("http://[fe80::1]:9999", 8080).unwrap();
        assert_eq!((e.host.as_str(), e.port), ("fe80::1", 9999));
        // Unbracketed: ambiguous, so the whole value is the host.
        let e = parse_host("::1", 1234).unwrap();
        assert_eq!((e.host.as_str(), e.port), ("::1", 1234));
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
        assert!(
            a.matches("bandwidth rather than").count() < 40,
            "too repetitive"
        );
    }

    /// A GGUF path and a catalog id are written differently and must still
    /// join. This is the only thing standing between llama.cpp and "cannot
    /// predict this model".
    #[test]
    fn a_gguf_filename_matches_its_catalog_entry() {
        let m = InstalledModel {
            name: "/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf".into(),
            size_bytes: 4_920_000_000,
            quant: "Q4_K_M".into(),
            ..Default::default()
        };
        let spec = catalog_match(&m).expect("should match llama-3.1-8b");
        assert_eq!(spec.id, "llama-3.1-8b");
        // Only the installed quantisation survives, at the real file's size.
        assert_eq!(spec.quants.len(), 1);
        assert_eq!(spec.quants[0].bytes, 4_920_000_000);
    }

    /// A quantisation nobody named cannot be predicted: the family sets
    /// dequantisation cost, and guessing it wrong moves decode by 30%.
    #[test]
    fn an_unlabelled_quantisation_is_refused() {
        let m = InstalledModel {
            name: "qwen3-4b".into(),
            size_bytes: 2_600_000_000,
            quant: "unknown".into(),
            ..Default::default()
        };
        assert!(catalog_match(&m).is_none());
    }

    /// A name match alone would pair a 70B file with the 8B entry. Every fact
    /// the runtime reported has to agree, or there is no match — predicting the
    /// wrong model is worse than declining to predict.
    #[test]
    fn a_disagreeing_parameter_count_rejects_the_match() {
        let m = InstalledModel {
            name: "llama-3.1-8b-Q4_K_M.gguf".into(),
            size_bytes: 4_920_000_000,
            quant: "Q4_K_M".into(),
            params: Some(70_600_000_000),
            ..Default::default()
        };
        assert!(catalog_match(&m).is_none());

        // The same model with a truthful count still matches.
        let ok = InstalledModel {
            params: Some(8_030_000_000),
            ..m
        };
        assert!(catalog_match(&ok).is_some());
    }
}
