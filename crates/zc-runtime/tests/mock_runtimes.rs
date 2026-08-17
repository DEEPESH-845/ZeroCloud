//! End-to-end tests against mock servers, one per runtime.
//!
//! Exercises the whole path — TCP, HTTP framing, JSON extraction, spec
//! construction, rate computation — without requiring any runtime to be
//! installed. Unit tests cover the parsers in isolation; this proves the pieces
//! actually fit together, which is the failure they cannot catch.
//!
//! Every payload here is a real response shape from that runtime's own
//! documentation. The numbers are chosen so the expected rate can be computed
//! by hand and asserted exactly.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use zc_runtime::{llamacpp::LlamaCpp, lmstudio::LmStudio, ollama::Ollama, openai::OpenAiCompat};
use zc_runtime::{Endpoint, InstalledModel, Runtime};

// -- Ollama -----------------------------------------------------------------

const OLLAMA_TAGS: &str = r#"{"models":[
{"name":"qwen3:4b","size":2600000000,"details":{"quantization_level":"Q4_K_M"}},
{"name":"qwen3:30b-a3b","size":18600000000,"details":{"quantization_level":"Q4_K_M"}}]}"#;

const OLLAMA_SHOW: &str = r#"{"model_info":{"general.architecture":"qwen3",
"general.parameter_count":4022468096,
"qwen3.block_count":36,"qwen3.embedding_length":2560,
"qwen3.attention.head_count":32,"qwen3.attention.head_count_kv":8,
"qwen3.attention.key_length":128,"qwen3.vocab_size":151936}}"#;

// 128 tokens / 2.28 s = 56.14 tok/s; 2048 / 1.2 s = 1706.67 tok/s.
const OLLAMA_GEN: &str = r#"{"model":"qwen3:4b","done":true,"done_reason":"length",
"total_duration":3521000000,"load_duration":41000000,
"prompt_eval_count":2048,"prompt_eval_duration":1200000000,
"eval_count":128,"eval_duration":2280000000}"#;

// -- llama.cpp --------------------------------------------------------------

const LLAMA_PROPS: &str = r#"{"default_generation_settings":{"id":0,"n_ctx":8192,
"params":{"n_predict":-1}},"total_slots":1,
"model_path":"../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
"build_info":"b4321-deadbeef"}"#;

const LLAMA_MODELS: &str = r#"{"object":"list","data":[{
"id":"../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf","object":"model",
"created":1735142223,"owned_by":"llamacpp",
"meta":{"vocab_type":2,"n_vocab":128256,"n_ctx_train":131072,
"n_embd":4096,"n_params":8030261312,"size":4912898304}}]}"#;

// 128 tokens / 4.0 s = 32 tok/s; 1200 / 0.5 s = 2400 tok/s.
const LLAMA_COMPLETION: &str = r#"{"content":"...","stop":true,
"timings":{"cache_n":0,"prompt_n":1200,"prompt_ms":500.0,
"predicted_n":128,"predicted_ms":4000.0}}"#;

// -- LM Studio --------------------------------------------------------------

const LMS_MODELS: &str = r#"{"object":"list","data":[
{"id":"qwen3-4b","object":"model","type":"llm","publisher":"qwen",
"arch":"qwen3","quantization":"Q4_K_M","state":"loaded","max_context_length":32768},
{"id":"nomic-embed-text","object":"model","type":"embeddings","quantization":"F16"}]}"#;

// 1200 prompt tokens / 0.5 s TTFT = 2400 tok/s prefill; decode stated directly.
const LMS_COMPLETION: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","model":"qwen3-4b",
"choices":[{"index":0,"finish_reason":"length","message":{"role":"assistant","content":"..."}}],
"usage":{"prompt_tokens":1200,"completion_tokens":128,"total_tokens":1328},
"stats":{"tokens_per_second":32.0,"time_to_first_token":0.5,"generation_time":4.0,
"stop_reason":"maxPredictedTokensReached"},
"model_info":{"arch":"qwen3","quant":"Q4_K_M","format":"gguf","context_length":8192}}"#;

const OPENAI_MODELS: &str = r#"{"object":"list","data":[
{"id":"meta-llama/Llama-3.1-8B-Instruct","object":"model","owned_by":"vllm"}]}"#;

/// Serve a fixed number of requests on an ephemeral port, routing on the
/// request path. Returns the port.
fn spawn(requests: usize, routes: &'static [(&'static str, &'static str)]) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();

    std::thread::spawn(move || {
        for stream in listener.incoming().take(requests) {
            let Ok(mut sock) = stream else { continue };
            let Ok(peek) = sock.try_clone() else { continue };
            let mut reader = BufReader::new(peek);

            // Read the request line and headers, noting any body length.
            let mut head = String::new();
            let mut body_len = 0usize;
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    body_len = v.trim().parse().unwrap_or(0);
                }
                if line == "\r\n" {
                    break;
                }
                head.push_str(&line);
            }
            if body_len > 0 {
                let mut buf = vec![0u8; body_len];
                let _ = reader.read_exact(&mut buf);
            }

            let payload = routes
                .iter()
                .find(|(path, _)| head.contains(path))
                .map(|(_, body)| *body)
                .unwrap_or("{}");
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                payload.len(),
                payload
            );
            let _ = sock.write_all(resp.as_bytes());
            let _ = sock.flush();
        }
    });
    port
}

fn ep(port: u16) -> Endpoint {
    Endpoint::local(port)
}

#[test]
fn ollama_round_trip() {
    let rt = Ollama::at(ep(spawn(
        4,
        &[
            ("/api/tags", OLLAMA_TAGS),
            ("/api/show", OLLAMA_SHOW),
            ("/api/generate", OLLAMA_GEN),
        ],
    )))
    .expect("health check should succeed");
    assert_eq!(rt.name(), "ollama");
    assert!(rt.calibratable());

    let models = rt.list().expect("list");
    assert_eq!(models.len(), 2);
    // Sorted largest first, so the 30B leads and the 4B is the default pick
    // for verify (smallest = fastest to measure).
    assert_eq!(models[0].name, "qwen3:30b-a3b");
    assert_eq!(models.last().unwrap().name, "qwen3:4b");

    // The one runtime that describes a model itself, so no catalog needed.
    let spec = rt
        .describe(&InstalledModel::new("qwen3:4b", 2_600_000_000, "Q4_K_M"))
        .expect("describe ok")
        .expect("spec present");
    assert_eq!(spec.n_layers, 36);
    // 2 * 36 layers * 8 kv heads * 128 dim * 2 bytes.
    assert_eq!(spec.kv_bytes(1, zc_model::KvPrecision::F16), 147_456);

    let run = rt.generate("qwen3:4b", 128, 4096, 42).expect("generate");
    assert_eq!(run.eval_tokens, 128);
    assert!((run.decode_tok_s - 56.14).abs() < 0.01, "{}", run.decode_tok_s);
    assert!((run.load_s - 0.041).abs() < 1e-6);
    // Ollama honours the requested context, so it has nothing to correct.
    assert_eq!(run.n_ctx, None);
}

/// llama.cpp reports timings but not geometry, so the spec has to come from
/// the catalog — and the match has to survive being checked against the facts
/// llama.cpp *did* report.
#[test]
fn llama_cpp_round_trip_joins_the_catalog() {
    let rt = LlamaCpp::at(ep(spawn(
        3,
        &[
            ("/props", LLAMA_PROPS),
            ("/v1/models", LLAMA_MODELS),
            ("/completion", LLAMA_COMPLETION),
        ],
    )))
    .expect("/props must identify llama.cpp");
    assert_eq!(rt.name(), "llamacpp");
    assert!(rt.calibratable());

    let models = rt.list().expect("list");
    assert_eq!(models[0].params, Some(8_030_261_312));

    // No geometry from the server; the catalog supplies it and every reported
    // fact is used to verify the join.
    assert!(rt.describe(&models[0]).expect("describe ok").is_none());
    let spec = zc_runtime::catalog_match(&models[0]).expect("catalog join");
    assert_eq!(spec.id, "llama-3.1-8b");
    // The real file's size wins over the catalog's representative figure.
    assert_eq!(spec.quants[0].bytes, 4_912_898_304);
    // 2 * 32 layers * 8 kv heads * 128 dim * 2 bytes.
    assert_eq!(spec.kv_bytes(1, zc_model::KvPrecision::F16), 131_072);

    // Milliseconds, not Ollama's nanoseconds: 128 / 4.0 s = 32 tok/s.
    let run = rt.generate(&spec.id, 128, 4096, 7).expect("generate");
    assert!((run.decode_tok_s - 32.0).abs() < 1e-6, "{}", run.decode_tok_s);
    assert!((run.prefill_tok_s - 2400.0).abs() < 1e-6);
    // The server's launch-time context, not the 4096 we asked for and it
    // ignored. Recording our request would put a fiction in the dataset.
    assert_eq!(run.n_ctx, Some(8192));
}

#[test]
fn lm_studio_round_trip_uses_the_native_stats_block() {
    let rt = LmStudio::at(ep(spawn(
        3,
        &[
            ("/api/v0/models", LMS_MODELS),
            ("/api/v0/chat/completions", LMS_COMPLETION),
        ],
    )))
    .expect("/api/v0/models must answer");
    assert_eq!(rt.name(), "lmstudio");
    assert!(rt.calibratable());

    // The embedding model is dropped: it cannot generate.
    let models = rt.list().expect("list");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "qwen3-4b");

    let spec = zc_runtime::catalog_match(&models[0]).expect("catalog join");
    assert_eq!(spec.id, "qwen3-4b");
    // LM Studio reports no size, so the catalog's own figure has to stand.
    assert!(spec.quants[0].bytes > 0);

    let run = rt.generate("qwen3-4b", 128, 4096, 7).expect("generate");
    assert!((run.decode_tok_s - 32.0).abs() < 1e-6);
    // 1200 prompt tokens against a 0.5 s time-to-first-token.
    assert!((run.prefill_tok_s - 2400.0).abs() < 1e-6);
    assert_eq!(run.n_ctx, Some(8192));
}

/// The rule the whole crate is built around: a runtime that cannot report its
/// own prefill/decode split must be visible but must never produce a number.
#[test]
fn an_openai_compatible_server_is_listed_but_cannot_calibrate() {
    let rt = OpenAiCompat::at(ep(spawn(2, &[("/v1/models", OPENAI_MODELS)])), "unknown")
        .expect("should be detected");
    // `owned_by` identifies it, not the port it happened to be on.
    assert_eq!(rt.name(), "vllm");
    assert!(!rt.calibratable());
    assert_eq!(rt.list().expect("list").len(), 1);

    let err = rt.generate("m", 128, 4096, 1).expect_err("must refuse");
    assert!(err.to_string().contains("prefill/decode"), "{err}");
}

#[test]
fn unreachable_endpoints_fail_cleanly() {
    // Port 1 on loopback: nothing listens, and this must not hang or panic.
    assert!(Ollama::at(ep(1)).is_none());
    assert!(LlamaCpp::at(ep(1)).is_none());
    assert!(LmStudio::at(ep(1)).is_none());
    assert!(OpenAiCompat::at(ep(1), "vllm").is_none());
}

/// The calibration inversion, driven by a mock's real numbers rather than a
/// hand-built struct: 56.14 tok/s measured against a raw time of 0.0125 s/token
/// implies eta = 0.702.
#[test]
fn calibration_inverts_a_measured_run_and_records_the_runtime() {
    let rt = Ollama::at(ep(spawn(2, &[("/api/tags", OLLAMA_TAGS), ("", OLLAMA_GEN)])))
        .expect("health");
    let run = rt.generate("qwen3:4b", 128, 4096, 7).expect("generate");

    let pred = zc_model::Prediction {
        resident_fraction: 1.0,
        decode_tok_s: (45.0, 75.0),
        prefill_tok_s: Some(20.0),
        ttft_s: Some(100.0),
        max_context: 32768,
        kv_bytes_per_token: 147_456,
        verdict: zc_model::Verdict::Good,
        raw_seconds_per_token: 0.0125,
        assumed_eta: 0.616,
        confidence: zc_model::Confidence::Prior,
        prefill_confidence: zc_model::Confidence::Prior,
    };

    let cal =
        zc_runtime::calibrate::compare("qwen3:4b", "Q4_K_M", &pred, &run, 4_022_468_096, 420.0);
    assert!(cal.within_range, "56.1 should fall inside 45-75");
    assert!((cal.implied_eta - 0.7018).abs() < 1e-3, "{}", cal.implied_eta);

    let line = zc_runtime::calibrate::record_line(
        "deadbeef", "macos", "none", "Metal", rt.name(), 132.0, 0.0, 5.0, 420.0, 4, "f16", 4096,
        4_022_468_096, &cal, &run,
    );
    assert!(!line.contains('\n'));
    assert!(line.contains(r#""actual_decode_tok_s":56.140"#), "{line}");
    // Prefill inversion must land in the record: 1706.67 * 2 * 4.022e9 / 4.2e11
    assert!(line.contains(r#""implied_prefill_scale":32.6"#), "{line}");
    // Which runtime produced the measurement is part of the record, because
    // runtimes differ in defaults and mixing them adds unexplained spread.
    assert!(line.contains(r#""runtime":"ollama""#), "{line}");
}

#[test]
fn quant_family_classification_is_exhaustive() {
    use zc_model::QuantFamily;
    for (label, want) in [
        ("Q4_K_M", QuantFamily::KQuant),
        ("Q8_0", QuantFamily::Legacy),
        ("IQ2_XXS", QuantFamily::IQuant),
        ("MXFP4", QuantFamily::BlockFloat),
        ("BF16", QuantFamily::Float),
    ] {
        assert_eq!(zc_runtime::ollama::quant_family(label), want, "{label}");
    }
}
