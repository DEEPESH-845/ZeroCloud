//! `zc serve` -- the same answers over HTTP, for tools that would rather ask
//! a port than parse a subprocess.
//!
//! The machine is measured once, before the first request, and every answer
//! is computed from that measurement: `/v1/check` is `zc check --json` and
//! `/v1/plan` is `zc plan --json`, through the same functions (see `api.rs`).
//!
//! Hand-written on `std::net`, like the HTTP client in `zc-runtime`, because
//! the dependency rule in CONTRIBUTING.md is the product. What that costs is
//! kept small on purpose:
//!
//! * **Loopback only.** There is no flag to bind elsewhere. Someone who wants
//!   it on a network can put a proxy in front and own that decision.
//! * **DNS rebinding is refused.** A web page can make a browser send requests
//!   to 127.0.0.1 under its own hostname; the `Host` header is the only thing
//!   that tells the two apart, so anything but a loopback name is a 403. No
//!   CORS header is ever sent, so a cross-origin page cannot read a response.
//! * **GET, one request per connection, 8 KiB of head, 5 s timeouts.** Nothing
//!   is read past the head, because no route takes a body.
//!
//! ponytail: one connection at a time. Every answer is arithmetic over a
//! measurement already taken, so a request costs microseconds; threads earn
//! their place only if a client ever holds a connection open on purpose.

use crate::{api, machine::Machine};
use std::io::{Read, Write};
use std::net::{Ipv4Addr, TcpListener};
use std::time::Duration;
use zc_model::Fit;

pub const DEFAULT_PORT: u16 = 8765;
const MAX_HEAD: usize = 8 * 1024;

pub fn bind(port: u16) -> std::io::Result<TcpListener> {
    TcpListener::bind((Ipv4Addr::LOCALHOST, port))
}

pub fn run(listener: TcpListener, m: &Machine, fit: &Fit) -> i32 {
    let addr = match listener.local_addr() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("zc serve: {e}");
            return 1;
        }
    };
    // stderr, and one line: a script waiting for readiness reads the port from
    // it (useful with --port 0), and stdout stays empty.
    eprintln!("zc serve: listening on http://{addr}  (GET /v1/check, /v1/plan?model=..., /health)");
    for conn in listener.incoming() {
        let Ok(mut s) = conn else { continue };
        let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
        let _ = s.set_write_timeout(Some(Duration::from_secs(5)));
        let answer = match read_head(&mut s).and_then(|h| route(&h)) {
            Ok(Route::Health) => Ok(format!(
                "{{\"ok\":true,\"version\":\"{}\"}}",
                env!("CARGO_PKG_VERSION")
            )),
            Ok(Route::Check(p)) => api::check(m, fit, &p),
            Ok(Route::Plan(p)) => api::plan(m, fit, &p),
            Err(f) => Err(f),
        };
        let (status, body) = answer.map_or_else(err, |b| (200, b));
        let _ = write_response(&mut s, status, &body);
    }
    0
}

fn err((status, msg): api::Failure) -> (u16, String) {
    (status, format!("{{\"error\":\"{}\"}}", zc_model::json::escape(&msg)))
}

#[derive(Debug, PartialEq)]
enum Route {
    Health,
    Check(Vec<(String, String)>),
    Plan(Vec<(String, String)>),
}

/// Read up to the blank line that ends the request head.
fn read_head(r: &mut impl Read) -> Result<String, api::Failure> {
    let mut buf = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
        if buf.len() > MAX_HEAD {
            return Err((431, "request head over 8 KiB".into()));
        }
        match r.read(&mut chunk) {
            Ok(0) | Err(_) => return Err((400, "incomplete request".into())),
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    }
    String::from_utf8(buf).map_err(|_| (400, "request head is not UTF-8".into()))
}

/// Method, Host and path checks, then the route. Pure, so it is testable
/// without a socket or a measured machine.
fn route(head: &str) -> Result<Route, api::Failure> {
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split(' ');
    let (method, target) = (first.next().unwrap_or(""), first.next().unwrap_or(""));
    let host = lines
        .take_while(|l| !l.is_empty())
        .filter_map(|l| l.split_once(':'))
        .find(|(k, _)| k.trim().eq_ignore_ascii_case("host"))
        .map(|(_, v)| v.trim());
    if let Some(h) = host.filter(|h| !host_is_loopback(h)) {
        return Err((403, format!("Host '{h}' is not a loopback name; zc serve answers only on this machine")));
    }
    if method != "GET" {
        return Err((405, format!("{method} is not supported; every route is GET")));
    }
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let params = parse_query(query)?;
    match path {
        "/health" | "/" => Ok(Route::Health),
        "/v1/check" => Ok(Route::Check(params)),
        "/v1/plan" => Ok(Route::Plan(params)),
        _ => Err((404, format!("no route {path} (try /v1/check, /v1/plan, /health)"))),
    }
}

/// `localhost:8765`, `127.0.0.1`, `[::1]:8765`. Anything else reached us
/// through a name that resolves here, which is what rebinding looks like.
fn host_is_loopback(host: &str) -> bool {
    let name = if host.starts_with('[') {
        host.split_once(']').map_or(host, |(h, _)| h).trim_start_matches('[')
    } else {
        host.split(':').next().unwrap_or(host)
    };
    matches!(name.to_ascii_lowercase().as_str(), "localhost" | "127.0.0.1" | "::1")
}

fn parse_query(q: &str) -> Result<Vec<(String, String)>, api::Failure> {
    q.split('&')
        .filter(|s| !s.is_empty())
        .map(|pair| {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            Ok((decode(k)?, decode(v)?))
        })
        .collect()
}

/// Percent-decoding, with `+` as a space.
fn decode(s: &str) -> Result<String, api::Failure> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' => {
                let hex = s.get(i + 1..i + 3).and_then(|h| u8::from_str_radix(h, 16).ok());
                let Some(byte) = hex else {
                    return Err((400, format!("bad percent-escape in '{s}'")));
                };
                out.push(byte);
                i += 2;
            }
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8(out).map_err(|_| (400, format!("'{s}' is not UTF-8 once decoded")))
}

fn write_response(w: &mut impl Write, status: u16, body: &str) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        431 => "Request Header Fields Too Large",
        _ => "Error",
    };
    let allow = if status == 405 { "Allow: GET\r\n" } else { "" };
    write!(
        w,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json; charset=utf-8\r\n\
         Content-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\
         {allow}Connection: close\r\n\r\n{body}\n",
        body.len() + 1
    )?;
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get(target: &str, host: &str) -> Result<Route, api::Failure> {
        route(&format!("GET {target} HTTP/1.1\r\nHost: {host}\r\nAccept: */*\r\n\r\n"))
    }

    #[test]
    fn routes_and_their_parameters() {
        assert_eq!(get("/health", "127.0.0.1:8765"), Ok(Route::Health));
        assert_eq!(
            get("/v1/plan?model=qwen3-8b&context=32K", "localhost:8765"),
            Ok(Route::Plan(vec![
                ("model".into(), "qwen3-8b".into()),
                ("context".into(), "32K".into())
            ]))
        );
        assert_eq!(
            get("/v1/check?model=llama%203&all_quants", "[::1]:8765"),
            Ok(Route::Check(vec![
                ("model".into(), "llama 3".into()),
                ("all_quants".into(), String::new())
            ]))
        );
        assert_eq!(get("/v2/nope", "localhost").unwrap_err().0, 404);
    }

    /// A page on evil.example resolving to 127.0.0.1 sends `Host:
    /// evil.example`. That header is the whole defence, so it is tested for
    /// the look-alikes too.
    #[test]
    fn a_rebound_hostname_is_refused() {
        for h in ["evil.example", "localhost.evil.example", "127.0.0.1.nip.io:8765", "[::2]:8765"] {
            assert_eq!(get("/health", h).unwrap_err().0, 403, "{h}");
        }
        assert!(host_is_loopback("LOCALHOST:1"));
    }

    #[test]
    fn only_get_is_served() {
        let e = route("POST /v1/check HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap_err();
        assert_eq!(e.0, 405);
    }

    #[test]
    fn a_head_that_never_ends_is_cut_off() {
        let mut endless = std::io::repeat(b'a');
        assert_eq!(read_head(&mut endless).unwrap_err().0, 431);
        let mut short: &[u8] = b"GET / HTTP/1.1\r\n";
        assert_eq!(read_head(&mut short).unwrap_err().0, 400);
    }

    #[test]
    fn percent_escapes_decode_or_fail_loudly() {
        assert_eq!(decode("a%2Bb+c").unwrap(), "a+b c");
        assert_eq!(decode("%zz").unwrap_err().0, 400);
        assert_eq!(decode("%").unwrap_err().0, 400);
        assert_eq!(decode("%ff").unwrap_err().0, 400);
    }
}
