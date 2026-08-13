//! End-to-end test against a mock Ollama server.
//!
//! Exercises the whole path — TCP, HTTP framing, JSON extraction, spec
//! construction, rate computation — without requiring a runtime to be
//! installed. Unit tests cover the parsers in isolation; this proves the
//! pieces actually fit together, which is the failure they cannot catch.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use zc_runtime::ollama::{self, Endpoint, InstalledModel};

const TAGS: &str = r#"{"models":[
{"name":"qwen3:4b","size":2600000000,"details":{"quantization_level":"Q4_K_M"}},
{"name":"qwen3:30b-a3b","size":18600000000,"details":{"quantization_level":"Q4_K_M"}}]}"#;

const SHOW: &str = r#"{"model_info":{"general.architecture":"qwen3",
"general.parameter_count":4022468096,
"qwen3.block_count":36,"qwen3.embedding_length":2560,
"qwen3.attention.head_count":32,"qwen3.attention.head_count_kv":8,
"qwen3.attention.key_length":128,"qwen3.vocab_size":151936}}"#;

// 128 tokens / 2.28 s = 56.14 tok/s; 2048 / 1.2 s = 1706.67 tok/s.
const GEN: &str = r#"{"model":"qwen3:4b","done":true,"done_reason":"length",
"total_duration":3521000000,"load_duration":41000000,
"prompt_eval_count":2048,"prompt_eval_duration":1200000000,
"eval_count":128,"eval_duration":2280000000}"#;

/// Serve a fixed number of requests on an ephemeral port, then stop.
fn spawn_mock(requests: usize) -> u16 {
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
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(_) => break,
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

            let payload = if head.contains("/api/tags") {
                TAGS
            } else if head.contains("/api/show") {
                SHOW
            } else {
                GEN
            };
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

fn endpoint(port: u16) -> Endpoint {
    Endpoint {
        host: "127.0.0.1".into(),
        port,
    }
}

#[test]
fn full_client_round_trip() {
    let ep = endpoint(spawn_mock(4));

    assert!(ep.is_running(), "health check should succeed");

    let models = ep.list().expect("list");
    assert_eq!(models.len(), 2);
    // Sorted largest first, so the 30B leads and the 4B is the default pick
    // for verify (smallest = fastest to measure).
    assert_eq!(models[0].name, "qwen3:30b-a3b");
    assert_eq!(models.last().unwrap().name, "qwen3:4b");

    let spec = ep
        .describe(&InstalledModel {
            name: "qwen3:4b".into(),
            size_bytes: 2_600_000_000,
            quant: "Q4_K_M".into(),
        })
        .expect("describe ok")
        .expect("spec present");
    assert_eq!(spec.n_layers, 36);
    assert_eq!(spec.n_embd, 2560);
    // 2 * 36 layers * 8 kv heads * 128 dim * 2 bytes.
    assert_eq!(spec.kv_bytes(1, zc_model::KvPrecision::F16), 147_456);

    let run = ep.generate("qwen3:4b", 128, 4096, 42).expect("generate");
    assert_eq!(run.eval_tokens, 128);
    assert!((run.decode_tok_s - 56.14).abs() < 0.01, "{}", run.decode_tok_s);
    assert!((run.load_s - 0.041).abs() < 1e-6);
}

#[test]
fn unreachable_endpoint_fails_cleanly() {
    // Port 1 on loopback: nothing listens, and this must not hang or panic.
    let ep = endpoint(1);
    assert!(!ep.is_running());
    assert!(ep.list().is_err());
    assert!(ep.generate("m", 8, 512, 1).is_err());
}

/// The calibration inversion, driven by the mock's real numbers rather than a
/// hand-built struct: 56.14 tok/s measured against a raw time of 0.0125 s/token
/// implies eta = 0.702.
#[test]
fn calibration_inverts_a_measured_run() {
    let ep = endpoint(spawn_mock(1));
    let run = ep.generate("qwen3:4b", 128, 4096, 7).expect("generate");

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

    let cal = zc_runtime::calibrate::compare(
        "qwen3:4b", "Q4_K_M", &pred, &run, 4_022_468_096, 420.0,
    );
    assert!(cal.within_range, "56.1 should fall inside 45-75");
    assert!((cal.implied_eta - 0.7018).abs() < 1e-3, "{}", cal.implied_eta);

    let line = zc_runtime::calibrate::record_line(
        "deadbeef", "macos", "none", "Metal", 132.0, 5.0, 420.0, 4, 4096, 4_022_468_096, &cal, &run,
    );
    assert!(!line.contains('\n'));
    assert!(line.contains(r#""actual_decode_tok_s":56.140"#), "{line}");
    // Prefill inversion must land in the record: 1706.67 * 2 * 4.022e9 / 4.2e11
    assert!(line.contains(r#""implied_prefill_scale":32.6"#), "{line}");
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
        assert_eq!(ollama::quant_family(label), want, "{label}");
    }
}
