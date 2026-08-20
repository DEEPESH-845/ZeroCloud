//! The parsers, against input this project did not write.
//!
//! Two of them see genuinely untrusted bytes: `json` parses Hugging Face
//! responses fetched over the network, and `Fit::from_text` parses calibration
//! records that arrive as community pull requests and are compiled into the
//! binary. A panic in either is a crash a stranger can cause, and `zc` is a
//! tool people run before they trust it.
//!
//! Deterministic rather than random so a failure is reproducible from the seed
//! alone. No fuzzing dependency: this is a loop and an xorshift.

/// Hostile shapes: truncations, unterminated strings, duplicate keys, numbers
/// that overflow every integer type, lone surrogates, deep nesting, a BOM.
const SEEDS: &[&str] = &[
    r#"{"a":1}"#,
    r#"{"a":"#,
    "{",
    "}",
    "",
    "null",
    "[]",
    r#"{"a":[[[[[[[[[[1]]]]]]]]]]}"#,
    r#"{"a":" \uD800"}"#,
    r#"{"n":999999999999999999999999999999}"#,
    r#"{"n":-0.0e999999}"#,
    r#"{"n":1e308}"#,
    r#"{"a":"unterminated"#,
    r#"{"a":{"b":{"c":{"d":{}}}}}"#,
    r#"{"num_hidden_layers":4294967296,"hidden_size":0,"vocab_size":-1}"#,
    r#"{"safetensors":{"parameters":{"BF16":1e308}}}"#,
    r#"{"a":"\\\\\\\\"}"#,
    r#"{"a":1,"a":2,"a":3}"#,
    r#"{"hw":"x","error_pct":"not a number","within_range":7}"#,
    r#"{"hw":null,"backend":[],"quant":{}}"#,
];

fn mutate(base: &str, r: u64) -> String {
    let mut s = base.to_string();
    match r % 6 {
        0 => {
            let n = (r as usize / 7) % (s.len() + 1);
            // Truncation must land on a character boundary, or `truncate`
            // panics on our own test input rather than on the parser.
            let n = (0..=n).rev().find(|i| s.is_char_boundary(*i)).unwrap_or(0);
            s.truncate(n);
        }
        1 => s.push_str(&"[".repeat((r as usize / 11) % 200)),
        2 => s.push_str(&"\u{1F600}".repeat((r as usize / 13) % 50)),
        3 => s.push('\u{0}'),
        4 => s.insert(0, '\u{feff}'),
        _ => s = s.repeat(1 + (r as usize / 17) % 3),
    }
    s
}

#[test]
fn no_hostile_input_panics_a_parser() {
    let mut state: u64 = 0x2545F4914F6CDD1D;
    let mut rnd = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for base in SEEDS {
        for _ in 0..500 {
            let s = mutate(base, rnd());
            // Every entry point that can see bytes we did not author.
            let _ = zc_model::json::number(&s, "a");
            let _ = zc_model::json::string(&s, "a");
            let _ = zc_model::json::boolean(&s, "a");
            let _ = zc_model::json::object_at(&s, "a");
            let _ = zc_model::json::array_objects(&s, "a");
            let _ = zc_model::json::number_by_suffix(&s, "a");
            let _ = zc_model::json::escape(&s);
            // Calibration records arrive as community pull requests.
            let _ = zc_model::Fit::from_text(&s);
        }
    }
}

/// A key that is not present, on input that is otherwise well-formed, must be
/// `None` rather than a wrong value from a neighbouring key.
#[test]
fn a_missing_key_is_absent_not_confused_with_a_neighbour() {
    let s = r#"{"aa":1,"ab":2,"b":3}"#;
    assert_eq!(zc_model::json::number(s, "a"), None);
    assert_eq!(zc_model::json::number(s, "aa"), Some(1.0));
    assert_eq!(zc_model::json::number(s, "ab"), Some(2.0));
    // A key that is a prefix of another must not match the longer one.
    let s = r#"{"error_pct_scaled":5,"error_pct":9}"#;
    assert_eq!(zc_model::json::number(s, "error_pct"), Some(9.0));
}

/// A number too large for the type it lands in must not wrap into a plausible
/// small value — that would be a silently wrong prediction rather than a
/// refusal.
#[test]
fn an_overflowing_number_does_not_wrap() {
    let huge = r#"{"n":999999999999999999999999999999}"#;
    let v = zc_model::json::number(huge, "n");
    assert!(
        v.is_none() || v.is_some_and(|x| x > 1e20),
        "a huge number came back as {v:?}"
    );
    let inf = r#"{"n":1e999}"#;
    let v = zc_model::json::number(inf, "n");
    assert!(
        v.is_none() || v.is_some_and(|x| !x.is_finite() || x > 1e300),
        "an overflowing exponent came back as {v:?}"
    );
}
