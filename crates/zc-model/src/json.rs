//! Minimal JSON value extraction.
//!
//! We talk to one localhost API with a known response shape and need perhaps
//! a dozen scalars from it. A full serde dependency would be the largest thing
//! in the binary by an order of magnitude, so instead this scans for a key and
//! parses the value that follows.
//!
//! ponytail: string-scanning extractor, not a parser. It is escape-aware and
//! will not match a key-shaped substring inside a string literal, which is the
//! failure that matters. It does NOT understand structure, so two keys with the
//! same name at different depths resolve to whichever comes first. Swap in a
//! real parser if we ever consume an API where that ambiguity is possible.

/// Byte offset just past `"<key>":`, skipping matches inside string literals.
fn find_key(src: &str, key: &str) -> Option<usize> {
    let b = src.as_bytes();
    let needle = format!("\"{key}\"");
    let n = needle.as_bytes();
    let mut i = 0;
    let mut in_string = false;

    while i < b.len() {
        if in_string {
            // Skip escapes so a `\"` does not look like the end of the string.
            if b[i] == b'\\' {
                i += 2;
                continue;
            }
            if b[i] == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if b[i] == b'"' {
            // A key always begins at a quote we are not already inside.
            if b[i..].starts_with(n) {
                let mut j = i + n.len();
                while j < b.len() && b[j].is_ascii_whitespace() {
                    j += 1;
                }
                if j < b.len() && b[j] == b':' {
                    j += 1;
                    while j < b.len() && b[j].is_ascii_whitespace() {
                        j += 1;
                    }
                    return Some(j);
                }
            }
            in_string = true;
        }
        i += 1;
    }
    None
}

/// Number at `"<key>"`. Accepts integers, decimals and exponents.
pub fn number(src: &str, key: &str) -> Option<f64> {
    let start = find_key(src, key)?;
    let rest = &src[start..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E')))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Boolean at `"<key>"`.
pub fn boolean(src: &str, key: &str) -> Option<bool> {
    let rest = &src[find_key(src, key)?..];
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

/// Decode the four hex digits of a `\uXXXX` escape sitting at `i`.
fn hex4(src: &str, i: usize) -> Option<u32> {
    u32::from_str_radix(src.get(i..i + 4)?, 16).ok()
}

/// String at `"<key>"`, with JSON escapes decoded.
///
/// Bytes are accumulated and decoded as UTF-8 at the end rather than pushed
/// individually as `char`. Pushing a raw byte as a `char` reinterprets it as a
/// code point, so every multi-byte character came out as mojibake — `é` became
/// `Ã©`. Qwen and other models carry non-ASCII GGUF metadata, so this was
/// reachable in ordinary use.
pub fn string(src: &str, key: &str) -> Option<String> {
    let start = find_key(src, key)?;
    let b = src.as_bytes();
    if b.get(start) != Some(&b'"') {
        return None;
    }
    let mut out: Vec<u8> = Vec::new();
    let mut i = start + 1;
    while i < b.len() {
        match b[i] {
            b'\\' => {
                i += 1;
                match b.get(i) {
                    Some(b'n') => {
                        out.push(b'\n');
                        i += 1;
                    }
                    Some(b't') => {
                        out.push(b'\t');
                        i += 1;
                    }
                    Some(b'r') => {
                        out.push(b'\r');
                        i += 1;
                    }
                    // Go's encoder — which is what Ollama uses — escapes `<`,
                    // `>` and `&` as </>/& by default, so this
                    // branch is hit by an ordinary response, not an exotic one.
                    // Undecoded, `&` arrived as the literal text `u0026`.
                    Some(b'u') => {
                        let Some(cp) = hex4(src, i + 1) else { break };
                        i += 5;
                        let ch = if (0xD800..0xDC00).contains(&cp) {
                            // High surrogate: a low surrogate must follow for a
                            // character outside the BMP.
                            let low = (src.as_bytes().get(i) == Some(&b'\\')
                                && src.as_bytes().get(i + 1) == Some(&b'u'))
                            .then(|| hex4(src, i + 2))
                            .flatten()
                            .filter(|l| (0xDC00..0xE000).contains(l));
                            match low {
                                Some(l) => {
                                    i += 6;
                                    char::from_u32(
                                        0x10000 + ((cp - 0xD800) << 10) + (l - 0xDC00),
                                    )
                                }
                                None => None,
                            }
                        } else {
                            char::from_u32(cp)
                        };
                        // A lone surrogate is not a character; emit U+FFFD
                        // rather than dropping it silently.
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(
                            ch.unwrap_or('\u{FFFD}').encode_utf8(&mut buf).as_bytes(),
                        );
                    }
                    Some(&c) => {
                        out.push(c);
                        i += 1;
                    }
                    None => break,
                }
            }
            b'"' => return Some(String::from_utf8_lossy(&out).into_owned()),
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    None
}

/// Slices of each object inside the array at `"<key>"`.
///
/// Tracks brace depth and string state rather than splitting on a literal like
/// `"},{"`, which silently returns one giant chunk the moment the server emits
/// whitespace between elements. Real Ollama sends compact JSON, so that bug
/// would have lain dormant until some proxy or future version pretty-printed.
pub fn array_objects<'a>(src: &'a str, key: &str) -> Vec<&'a str> {
    let Some(start) = find_key(src, key) else {
        return Vec::new();
    };
    let b = src.as_bytes();
    if b.get(start) != Some(&b'[') {
        return Vec::new();
    }
    let mut out = Vec::new();
    let (mut i, mut depth, mut obj_start) = (start + 1, 0usize, 0usize);
    let mut in_string = false;

    while i < b.len() {
        if in_string {
            if b[i] == b'\\' {
                i += 2;
                continue;
            }
            if b[i] == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        match b[i] {
            b'"' => in_string = true,
            b'{' => {
                if depth == 0 {
                    obj_start = i;
                }
                depth += 1;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    out.push(&src[obj_start..=i]);
                }
            }
            b']' if depth == 0 => break,
            _ => {}
        }
        i += 1;
    }
    out
}

/// The object value at `"<key>"`, as a slice including its braces.
///
/// Needed for catalog files, where `attention` and `moe` are nested objects
/// rather than flat scalars. Shares the depth/string tracking with
/// `array_objects`, so a brace inside a string value cannot end it early.
pub fn object_at<'a>(src: &'a str, key: &str) -> Option<&'a str> {
    let start = find_key(src, key)?;
    let b = src.as_bytes();
    if b.get(start) != Some(&b'{') {
        return None;
    }
    let (mut i, mut depth) = (start, 0usize);
    let mut in_string = false;
    while i < b.len() {
        if in_string {
            if b[i] == b'\\' {
                i += 2;
                continue;
            }
            if b[i] == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        match b[i] {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&src[start..=i]);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Number for the first key ending in `suffix`.
///
/// GGUF metadata namespaces its keys by architecture — `qwen3.block_count`,
/// `llama.block_count` — so we match on the suffix rather than hardcoding
/// every architecture name.
pub fn number_by_suffix(src: &str, suffix: &str) -> Option<f64> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut in_string = false;
    while i < b.len() {
        if in_string {
            if b[i] == b'\\' {
                i += 2;
                continue;
            }
            if b[i] == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if b[i] == b'"' {
            if let Some(close) = src[i + 1..].find('"') {
                let key = &src[i + 1..i + 1 + close];
                if key.ends_with(suffix)
                    && let Some(v) = number(src, key)
                {
                    return Some(v);
                }
                i += close + 2;
                continue;
            }
            in_string = true;
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trimmed from a real `/api/generate` response.
    const GEN: &str = r#"{"model":"qwen3:4b","created_at":"2026-08-13T00:00:00Z",
        "response":"Hello","done":true,"done_reason":"stop",
        "context":[151644,872,198,9707],
        "total_duration":3521000000,"load_duration":41000000,
        "prompt_eval_count":2048,"prompt_eval_duration":1200000000,
        "eval_count":128,"eval_duration":2280000000}"#;

    #[test]
    fn extracts_scalars() {
        assert_eq!(string(GEN, "model").as_deref(), Some("qwen3:4b"));
        assert_eq!(number(GEN, "eval_count"), Some(128.0));
        assert_eq!(number(GEN, "eval_duration"), Some(2_280_000_000.0));
    }

    /// `"prompt_eval_count"` contains the text `eval_count`, so a naive
    /// substring search would return the wrong field. The leading quote in the
    /// needle is what prevents that — this test locks the behaviour in.
    #[test]
    fn does_not_confuse_prefixed_keys() {
        assert_eq!(number(GEN, "prompt_eval_count"), Some(2048.0));
        assert_eq!(number(GEN, "eval_count"), Some(128.0));
        assert_ne!(number(GEN, "eval_count"), number(GEN, "prompt_eval_count"));
    }

    /// A key-shaped substring inside a string value must not match.
    #[test]
    fn ignores_keys_inside_string_values() {
        let s = r#"{"response":"the \"eval_count\": 999 trick","eval_count":7}"#;
        assert_eq!(number(s, "eval_count"), Some(7.0));
    }

    #[test]
    fn suffix_match_handles_architecture_prefixes() {
        let s = r#"{"model_info":{"general.architecture":"qwen3",
            "qwen3.block_count":36,"qwen3.attention.head_count_kv":8,
            "qwen3.attention.key_length":128,"qwen3.embedding_length":2560}}"#;
        assert_eq!(number_by_suffix(s, ".block_count"), Some(36.0));
        assert_eq!(number_by_suffix(s, ".attention.head_count_kv"), Some(8.0));
        assert_eq!(number_by_suffix(s, ".embedding_length"), Some(2560.0));
    }

    /// Whitespace between array elements must not collapse them into one.
    #[test]
    fn splits_array_objects_regardless_of_formatting() {
        let compact = r#"{"models":[{"name":"a","size":1},{"name":"b","size":2}]}"#;
        let pretty = "{\"models\":[\n  {\"name\":\"a\",\"size\":1},\n  {\"name\":\"b\",\"size\":2}\n]}";
        for src in [compact, pretty] {
            let v = array_objects(src, "models");
            assert_eq!(v.len(), 2, "{src}");
            assert_eq!(string(v[0], "name").as_deref(), Some("a"));
            assert_eq!(string(v[1], "name").as_deref(), Some("b"));
        }
    }

    /// Nested objects must not be mistaken for element boundaries.
    #[test]
    fn splits_array_objects_with_nesting() {
        let s = r#"{"models":[{"name":"a","details":{"q":"Q4"}},{"name":"b","details":{"q":"Q8"}}]}"#;
        let v = array_objects(s, "models");
        assert_eq!(v.len(), 2);
        assert_eq!(string(v[1], "q").as_deref(), Some("Q8"));
    }

    /// A brace inside a string value must not shift the depth counter.
    #[test]
    fn braces_inside_strings_do_not_break_splitting() {
        let s = r#"{"models":[{"name":"a{b}c"},{"name":"d"}]}"#;
        let v = array_objects(s, "models");
        assert_eq!(v.len(), 2);
        assert_eq!(string(v[0], "name").as_deref(), Some("a{b}c"));
    }

    #[test]
    fn extracts_nested_objects() {
        let s = r#"{"id":"m","attention":{"kind":"gqa","n_kv_heads":8},"moe":null}"#;
        let a = object_at(s, "attention").expect("attention");
        assert_eq!(string(a, "kind").as_deref(), Some("gqa"));
        assert_eq!(number(a, "n_kv_heads"), Some(8.0));
        // A null value is not an object and must not be mistaken for one.
        assert!(object_at(s, "moe").is_none());
    }

    /// The extracted slice must stop at its own closing brace, not the outer
    /// document's, even when the object contains further nesting.
    #[test]
    fn nested_object_is_bounded_correctly() {
        let s = r#"{"a":{"b":{"c":1},"d":2},"e":{"f":3}}"#;
        let a = object_at(s, "a").expect("a");
        assert_eq!(a, r#"{"b":{"c":1},"d":2}"#);
        assert_eq!(number(a, "f"), None);
    }

    /// Ollama is written in Go, whose JSON encoder escapes `<`, `>` and `&` by
    /// default. These arrive on ordinary responses, and were previously
    /// returned as the literal text `u0026`.
    #[test]
    fn decodes_unicode_escapes_go_emits() {
        let s = r#"{"name":"a&b","tpl":"<|im_start|>"}"#;
        assert_eq!(string(s, "name").as_deref(), Some("a&b"));
        assert_eq!(string(s, "tpl").as_deref(), Some("<|im_start|>"));
    }

    /// Multi-byte UTF-8 must survive intact. Pushing each byte as a `char`
    /// turned "é" into "Ã©" and corrupted every non-ASCII model name.
    #[test]
    fn multibyte_utf8_is_not_mangled() {
        let s = r#"{"name":"café","cn":"通义千问"}"#;
        assert_eq!(string(s, "name").as_deref(), Some("café"));
        assert_eq!(string(s, "cn").as_deref(), Some("通义千问"));
    }

    /// Surrogate pairs form one character; a lone surrogate is not a character
    /// at all and must not silently vanish.
    #[test]
    fn handles_surrogate_pairs_and_lone_surrogates() {
        assert_eq!(string(r#"{"e":"😀"}"#, "e").as_deref(), Some("😀"));
        assert_eq!(string(r#"{"e":"\uD83D"}"#, "e").as_deref(), Some("\u{FFFD}"));
    }

    /// A truncated escape must terminate rather than read past the end.
    #[test]
    fn truncated_escape_does_not_panic() {
        assert_eq!(string(r#"{"e":"\u00"#, "e"), None);
        assert_eq!(string(r#"{"e":"ab\"#, "e"), None);
    }

    #[test]
    fn missing_keys_are_none() {
        assert_eq!(number(GEN, "nope"), None);
        assert_eq!(string(GEN, "nope"), None);
    }
}
