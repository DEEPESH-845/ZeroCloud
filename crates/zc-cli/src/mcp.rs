//! `zc serve --mcp` -- the same two answers as tools an AI agent can call.
//!
//! Model Context Protocol over stdio: one JSON-RPC 2.0 message per line in,
//! one per line out, nothing else on stdout. Two tools, `check` and `plan`,
//! each returning exactly the JSON `zc check --json` / `zc plan --json` print.
//!
//! Both protocol generations are spoken, because clients in use today are
//! split between them:
//!
//! * up to 2025-11-25, a session opens with `initialize`;
//! * 2026-07-28 is stateless -- `server/discover` replaces the handshake and
//!   every request names its version in `_meta`, which is checked here and
//!   refused with -32022 when unsupported.
//!
//! The server keeps no session either way, so the difference is two methods.
//! The machine is measured on the first tool call rather than at start-up:
//! a client waits for `initialize` before it will do anything, and the
//! benchmark takes seconds to a minute.

use crate::{api, machine::Machine};
use std::io::{BufRead, Write};
use zc_model::json::{escape, field, members, unquote};
use zc_model::Fit;

const SUPPORTED: &[&str] = &["2026-07-28", "2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];
/// The newest revision that still has an `initialize` handshake.
const HANDSHAKE_LATEST: &str = "2025-11-25";

const INSTRUCTIONS: &str = "zc measures this machine (memory bandwidth, compute, disk) and \
predicts how local LLMs will run on it. Use `check` to see what fits and how fast, and \
`plan` to ask what one model would need. Numbers are ranges with a stated confidence; \
a null is unmeasured, never zero. The first call benchmarks the machine.";

const TOOLS: &str = r#"{"tools":[
{"name":"check","title":"What can this machine run?",
"description":"Measured hardware plus a ranked prediction per catalog model: verdict (good, usable, slow, wont_fit), decode tokens/s as a low-high range, max context, time to first token (null until measured), and confidence. The first call benchmarks this machine, which takes a few seconds to about a minute; later calls reuse that measurement.",
"inputSchema":{"type":"object","properties":{
"model":{"type":"string","description":"Only models whose catalog id contains this, e.g. qwen3"},
"top":{"type":"integer","minimum":1,"description":"Row limit after ranking. Default 20; unlimited with all_quants unless given."},
"all_quants":{"type":"boolean","description":"Every quantisation rather than the best one per model."},
"kv":{"type":"string","enum":["f16","q8","q4"],"description":"KV-cache precision. Default f16, which every runtime ships."}},
"additionalProperties":false},
"annotations":{"readOnlyHint":true,"openWorldHint":false}},
{"name":"plan","title":"What would this model need?",
"description":"For one catalog model: memory per quantisation at a context length, and the memory bandwidth needed for a target decode rate, judged against this machine's measured budget.",
"inputSchema":{"type":"object","properties":{
"model":{"type":"string","description":"Catalog id or an unambiguous part of one, e.g. qwen3-8b"},
"context":{"type":["integer","string"],"description":"Context in tokens, e.g. 4096 or \"32K\". Default 4096."},
"quant":{"type":"string","description":"One quantisation, e.g. Q4_K_M. Default: all of them."},
"kv":{"type":"string","enum":["f16","q8","q4"]},
"target_tps":{"type":"number","exclusiveMinimum":0,"description":"Target decode rate. Default 10 tokens/s."}},
"required":["model"],"additionalProperties":false},
"annotations":{"readOnlyHint":true,"openWorldHint":false}}]}"#;

pub fn run(fit: &Fit) -> i32 {
    let mut machine: Option<Machine> = None;
    let mut out = std::io::stdout().lock();
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let reply = handle(&line, &mut |tool, params| {
            let m = machine.get_or_insert_with(crate::machine::probe);
            match tool {
                "check" => api::check(m, fit, params),
                _ => api::plan(m, fit, params),
            }
        });
        if let Some(r) = reply
            && writeln!(out, "{r}").and_then(|_| out.flush()).is_err()
        {
            break;
        }
    }
    0
}

type Call<'a> = dyn FnMut(&str, &[(String, String)]) -> Result<String, api::Failure> + 'a;

/// One message in, at most one out. `None` for a notification, or anything
/// else that must not be answered.
fn handle(line: &str, call: &mut Call) -> Option<String> {
    let msg = line.trim();
    if !msg.starts_with('{') || members(msg).is_empty() {
        return Some(error("null", -32700, "parse error: expected one JSON-RPC object per line"));
    }
    let id = field(msg, "id");
    let Some(method) = field(msg, "method").and_then(unquote) else {
        // A response to a request we never sent, or garbage. Only a message
        // with an id can be answered, and only with an error.
        let is_response = field(msg, "result").is_some() || field(msg, "error").is_some();
        return id.filter(|_| !is_response).map(|id| error(id, -32600, "invalid request: no method"));
    };
    let params = field(msg, "params").unwrap_or("{}");

    let requested = field(params, "_meta")
        .and_then(|m| field(m, "io.modelcontextprotocol/protocolVersion"))
        .and_then(unquote);
    if let Some(v) = requested.filter(|v| !SUPPORTED.contains(&v.as_str())) {
        return id.map(|id| unsupported(id, &v));
    }
    let id = id?;

    let result = match method.as_str() {
        "initialize" => {
            let asked = field(params, "protocolVersion").and_then(unquote);
            let version = match asked.as_deref() {
                Some(v) if SUPPORTED.contains(&v) && v != "2026-07-28" => v,
                _ => HANDSHAKE_LATEST,
            };
            format!(
                "{{\"protocolVersion\":\"{version}\",\"capabilities\":{{\"tools\":{{}}}},\
                 \"serverInfo\":{{\"name\":\"zerocloud\",\"version\":\"{}\"}},\"instructions\":\"{}\"}}",
                env!("CARGO_PKG_VERSION"),
                escape(INSTRUCTIONS)
            )
        }
        "server/discover" => format!(
            "{{\"supportedVersions\":[{}],\"capabilities\":{{\"tools\":{{}}}},\"instructions\":\"{}\"}}",
            SUPPORTED.iter().map(|v| format!("\"{v}\"")).collect::<Vec<_>>().join(","),
            escape(INSTRUCTIONS)
        ),
        "ping" => "{}".into(),
        "tools/list" => TOOLS.replace('\n', ""),
        "tools/call" => {
            let name = field(params, "name").and_then(unquote);
            let tool = match name.as_deref() {
                Some(t @ ("check" | "plan")) => t,
                Some(other) => return Some(error(id, -32602, &format!("unknown tool '{other}'"))),
                None => return Some(error(id, -32602, "tools/call needs a tool name")),
            };
            let args: Vec<(String, String)> = members(field(params, "arguments").unwrap_or("{}"))
                .into_iter()
                .map(|(k, raw)| (k, unquote(raw).unwrap_or_else(|| raw.to_string())))
                .collect();
            // A bad argument or an unknown model is the tool failing, not the
            // protocol: the spec puts it in the result, flagged, so the model
            // sees the message and can correct itself.
            let (text, is_error) = match call(tool, &args) {
                Ok(json) => (json, false),
                Err((_, msg)) => (msg, true),
            };
            format!(
                "{{\"content\":[{{\"type\":\"text\",\"text\":\"{}\"}}],\"isError\":{is_error}}}",
                escape(&text)
            )
        }
        other => return Some(error(id, -32601, &format!("method not found: {other}"))),
    };
    Some(format!("{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":{result}}}"))
}

fn error(id: &str, code: i32, msg: &str) -> String {
    format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"error\":{{\"code\":{code},\"message\":\"{}\"}}}}",
        escape(msg)
    )
}

fn unsupported(id: &str, requested: &str) -> String {
    format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"error\":{{\"code\":-32022,\
         \"message\":\"unsupported protocol version\",\"data\":{{\"supported\":[{}],\"requested\":\"{}\"}}}}}}",
        SUPPORTED.iter().map(|v| format!("\"{v}\"")).collect::<Vec<_>>().join(","),
        escape(requested)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in for the measured machine: echoes what it was asked.
    fn echo(tool: &str, args: &[(String, String)]) -> Result<String, api::Failure> {
        if args.iter().any(|(_, v)| v == "missing") {
            return Err((404, "no catalog model matches 'missing'".into()));
        }
        Ok(format!("{tool}:{args:?}"))
    }

    fn ask(line: &str) -> Option<String> {
        handle(line, &mut echo)
    }

    #[test]
    fn the_handshake_echoes_a_supported_version_and_falls_back_otherwise() {
        let r = ask(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"1"}}}"#).unwrap();
        assert!(r.contains(r#""protocolVersion":"2025-06-18""#), "{r}");
        assert!(r.contains(r#""id":1,"#));
        let r = ask(r#"{"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"2099-01-01"}}"#).unwrap();
        assert!(r.contains(r#""protocolVersion":"2025-11-25""#), "{r}");
    }

    /// 2026-07-28 has no handshake: discovery, then self-describing requests.
    #[test]
    fn the_stateless_revision_is_spoken_and_checked() {
        let r = ask(r#"{"jsonrpc":"2.0","id":"d","method":"server/discover","params":{}}"#).unwrap();
        assert!(r.contains(r#""supportedVersions":["2026-07-28""#), "{r}");
        let meta = |v: &str| {
            format!(r#"{{"jsonrpc":"2.0","id":3,"method":"tools/list","params":{{"_meta":{{"io.modelcontextprotocol/protocolVersion":"{v}","io.modelcontextprotocol/clientCapabilities":{{}}}}}}}}"#)
        };
        assert!(ask(&meta("2026-07-28")).unwrap().contains(r#""name":"check""#));
        let r = ask(&meta("2030-01-01")).unwrap();
        assert!(r.contains("-32022") && r.contains(r#""requested":"2030-01-01""#), "{r}");
    }

    #[test]
    fn tool_calls_pass_arguments_and_report_tool_errors_in_the_result() {
        let r = ask(r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"arguments":{"model":"qwen3-8b","context":"32K","target_tps":20},"name":"plan"}}"#).unwrap();
        // Arguments before name, and a string and a number both arrive as text.
        assert!(r.contains(r#"plan:[(\"model\", \"qwen3-8b\"), (\"context\", \"32K\"), (\"target_tps\", \"20\")]"#), "{r}");
        assert!(r.contains(r#""isError":false"#));
        let r = ask(r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"check","arguments":{"model":"missing"}}}"#).unwrap();
        assert!(r.contains(r#""isError":true"#) && r.contains("no catalog model"), "{r}");
        let r = ask(r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"rm"}}"#).unwrap();
        assert!(r.contains("-32602"), "{r}");
    }

    #[test]
    fn notifications_and_stray_responses_get_no_reply() {
        assert_eq!(ask(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#), None);
        assert_eq!(ask(r#"{"jsonrpc":"2.0","id":7,"result":{}}"#), None);
        assert!(ask("not json").unwrap().contains("-32700"));
        assert!(ask(r#"{"jsonrpc":"2.0","id":8,"method":"resources/list"}"#).unwrap().contains("-32601"));
    }

    /// The tool list is a hand-written constant, so prove it is one JSON
    /// object with both tools rather than trusting the eye.
    #[test]
    fn the_tool_list_is_well_formed() {
        let flat = TOOLS.replace('\n', "");
        let tools = zc_model::json::array_objects(&flat, "tools");
        let names: Vec<_> = tools.iter().filter_map(|t| field(t, "name").and_then(unquote)).collect();
        assert_eq!(names, ["check", "plan"]);
        for t in tools {
            assert!(field(t, "inputSchema").is_some_and(|s| !members(s).is_empty()));
        }
    }
}
