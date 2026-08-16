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
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// Generation on a low-end machine with a large model is genuinely slow, and
/// timing out mid-measurement wastes the whole run.
const READ_TIMEOUT: Duration = Duration::from_secs(600);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
/// Every response we consume is metadata or a 128-token completion — kilobytes.
/// This ceiling is far above anything legitimate and still bounds the damage a
/// hostile or malfunctioning endpoint can do.
const MAX_RESPONSE_BYTES: u64 = 64 << 20;

pub struct Response {
    pub status: u16,
    pub body: String,
}

fn request(host: &str, port: u16, head: &str, body: Option<&str>) -> std::io::Result<Response> {
    // Resolve rather than parse. `SocketAddr::from_str` accepts only IP
    // literals, so the previous `format!("{host}:{port}").parse()` rejected
    // `OLLAMA_HOST=localhost:11434` outright — the most natural way to write
    // it, and what Ollama's own docs use. The tuple form also sidesteps IPv6
    // bracket quoting, which string concatenation gets wrong.
    let addrs: Vec<_> = (host, port).to_socket_addrs()?.collect();

    // `localhost` resolves to both ::1 and 127.0.0.1, and a runtime bound only
    // to IPv4 refuses the v6 address. Trying just the first would fail against
    // a perfectly healthy server, so walk them and keep the last error.
    let mut stream = None;
    let mut last_err = None;
    for addr in &addrs {
        match TcpStream::connect_timeout(addr, CONNECT_TIMEOUT) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(e) => last_err = Some(e),
        }
    }
    let mut stream = match stream {
        Some(s) => s,
        None => {
            return Err(last_err.unwrap_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("{host}:{port} resolved to no addresses"),
                )
            }))
        }
    };
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

    // `OLLAMA_HOST` points wherever the user says, so this is not necessarily
    // a trusted localhost process. An unbounded `read_to_end` lets whatever is
    // on the far end grow our heap until the machine dies -- on a tool whose
    // whole audience is memory-constrained laptops.
    let mut raw = Vec::new();
    stream.take(MAX_RESPONSE_BYTES).read_to_end(&mut raw)?;
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
/// Lives in `zc_model::json` alongside the reader, so writers everywhere share
/// one implementation. Re-exported here because this is where callers expect it.
pub use zc_model::json::escape as json_escape;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_json_specials() {
        assert_eq!(json_escape(r#"a"b\c"#), r#"a\"b\\c"#);
        assert_eq!(json_escape("a\nb"), "a\\nb");
        assert_eq!(json_escape("a\u{1}b"), "a\\u0001b");
    }

    /// `OLLAMA_HOST=localhost:11434` used to fail with "bad address" because
    /// `SocketAddr::from_str` accepts only IP literals. The listener binds
    /// IPv4-only while `localhost` usually resolves to ::1 first, so this also
    /// covers walking past an address the runtime is not bound to.
    #[test]
    fn connects_to_a_hostname_not_just_an_ip() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();

        std::thread::spawn(move || {
            for conn in listener.incoming().take(1) {
                let mut c = conn.expect("accept");
                let mut sink = [0u8; 1024];
                let _ = c.read(&mut sink);
                let _ = c.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"ok\":true}");
            }
        });

        let r = get("localhost", port, "/api/version").expect("hostname must resolve");
        assert_eq!(r.status, 200);
        assert_eq!(r.body, r#"{"ok":true}"#);
    }

    #[test]
    fn connect_failure_is_an_error_not_a_panic() {
        // Nothing listens on this port; must fail cleanly.
        assert!(get("127.0.0.1", 1, "/").is_err());
    }
}
