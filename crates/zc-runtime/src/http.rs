//! A localhost-only HTTP/1.1 client.
//!
//! We make exactly three requests to a service on 127.0.0.1. Pulling in a
//! full HTTP stack (and with it an async runtime) for that would dwarf the
//! rest of the binary.
//!
//! We send `Connection: close` so the server ends the stream when the body is
//! done, which lets us read to EOF and skip Content-Length and chunked
//! decoding entirely. That trade is only acceptable because we are not
//! reusing connections.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// Generation on a low-end machine with a large model is genuinely slow, and
/// timing out mid-measurement wastes the whole run.
const READ_TIMEOUT: Duration = Duration::from_secs(600);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);

pub struct Response {
    pub status: u16,
    pub body: String,
}

fn request(host: &str, port: u16, head: &str, body: Option<&str>) -> std::io::Result<Response> {
    let addr = format!("{host}:{port}")
        .parse()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "bad address"))?;
    let mut stream = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)?;
    stream.set_read_timeout(Some(READ_TIMEOUT))?;
    // Small JSON requests; Nagle would add 40ms of pointless latency.
    stream.set_nodelay(true)?;

    let mut req = String::from(head);
    req.push_str(&format!("Host: {host}:{port}\r\nConnection: close\r\n"));
    if let Some(b) = body {
        req.push_str("Content-Type: application/json\r\n");
        req.push_str(&format!("Content-Length: {}\r\n", b.len()));
    }
    req.push_str("\r\n");
    if let Some(b) = body {
        req.push_str(b);
    }
    stream.write_all(req.as_bytes())?;
    stream.flush()?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    let text = String::from_utf8_lossy(&raw).into_owned();

    let status = text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    // Headers and body are separated by a blank line.
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();

    Ok(Response { status, body })
}

pub fn get(host: &str, port: u16, path: &str) -> std::io::Result<Response> {
    request(host, port, &format!("GET {path} HTTP/1.1\r\n"), None)
}

pub fn post_json(host: &str, port: u16, path: &str, body: &str) -> std::io::Result<Response> {
    request(host, port, &format!("POST {path} HTTP/1.1\r\n"), Some(body))
}

/// Escape a string for embedding in a JSON literal.
///
/// Only needed for the prompt we send. Control characters below 0x20 must be
/// escaped or the server rejects the document.
pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_json_specials() {
        assert_eq!(json_escape(r#"a"b\c"#), r#"a\"b\\c"#);
        assert_eq!(json_escape("a\nb"), "a\\nb");
        assert_eq!(json_escape("a\u{1}b"), "a\\u0001b");
    }

    #[test]
    fn connect_failure_is_an_error_not_a_panic() {
        // Nothing listens on this port; must fail cleanly.
        assert!(get("127.0.0.1", 1, "/").is_err());
    }
}
