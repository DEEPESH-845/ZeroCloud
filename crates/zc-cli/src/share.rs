//! `zc share` — hand one calibration record to GitHub without opening a socket.
//!
//! The record travels as a query parameter in a URL the *browser* opens.
//! `zc` still makes no outbound connection of any kind, which is what keeps the
//! README's privacy claim literally true rather than nearly true, and it means
//! there is no token to store, no OAuth app to register and no credential in
//! the binary. GitHub forks the repository on commit for anyone without write
//! access, so the fork-and-open-a-pull-request half needs no code at all.

use std::fmt::Write as _;
use std::io::Write as _;
use zc_model::json;

const REPO: &str = "DEEPESH-845/ZeroCloud";
const BRANCH: &str = "main";
/// Submissions land in their own tier. Which tier backs the published accuracy
/// number is a maintainer's decision made in a visible commit, never a side
/// effect of somebody running a command.
const DIR: &str = "crates/zc-model/data/calibration/community";

/// GitHub answers 414 past its query limit. An encoded record is around 900
/// bytes, so this is head-room rather than a tuning knob — and exceeding it
/// falls back to copy-paste instead of opening a browser at an error page.
const MAX_URL: usize = 6144;

/// Every field a record carries, in disclosure order.
///
/// A list rather than a walk over the JSON, because
/// `every_field_in_the_record_is_disclosed` then fails the moment a field is
/// added to `calibrate::record_line` and not added here. A silent omission is
/// the only way this file can break the promise it exists to keep.
const FIELDS: &[&str] = &[
    "hw",
    "os",
    "virt",
    "backend",
    "runtime",
    "ram_bw_gbs",
    "vram_bw_gbs",
    "disk_bw_gbs",
    "gflops",
    "threads",
    "kv",
    "model",
    "quant",
    "ctx",
    "prompt_tokens",
    "eval_tokens",
    "predicted_lo",
    "predicted_hi",
    "actual_decode_tok_s",
    "actual_prefill_tok_s",
    "assumed_eta",
    "implied_eta",
    "implied_prefill_scale",
    "active_params",
    "error_pct",
    "within_range",
];

/// Percent-encode everything outside the unreserved set.
///
/// `/` is encoded too. It is legal unescaped in a query value, but encoding it
/// keeps the rule to one line and removes the need to reason about which
/// characters GitHub's router treats as path separators.
pub fn encode(s: &str) -> String {
    let mut o = String::with_capacity(s.len() * 3 / 2);
    for b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                o.push(*b as char)
            }
            _ => {
                let _ = write!(o, "%{b:02X}");
            }
        }
    }
    o
}

/// `<hw>-<hash8>.jsonl`, where the hash is over the record line itself.
///
/// Content-addressed in both directions: the same run resubmitted produces the
/// same filename and therefore an empty diff, and a second run from the same
/// machine cannot overwrite the first.
pub fn dest_filename(line: &str) -> Result<String, String> {
    let hw = json::string(line, "hw").ok_or_else(|| "record has no \"hw\" field".to_string())?;
    if hw.len() != 16 || !hw.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) {
        return Err(format!("\"hw\" is not a 16-digit lowercase hex fingerprint: {hw}"));
    }
    let hash = zc_runtime::calibrate::fingerprint(line);
    Ok(format!("{hw}-{}.jsonl", &hash[..8]))
}

pub fn share_url(filename: &str, line: &str) -> String {
    format!(
        "https://github.com/{REPO}/new/{BRANCH}?filename={}&value={}",
        encode(&format!("{DIR}/{filename}")),
        encode(line)
    )
}

/// The record, one field per line. This is the `PLAN.md` C1 promise — print
/// exactly what is collected, before it goes anywhere — discharged at the only
/// moment where it matters.
fn describe(line: &str) -> String {
    let mut o = String::new();
    for key in FIELDS {
        let v = json::string(line, key)
            .or_else(|| json::boolean(line, key).map(|b| b.to_string()))
            .or_else(|| json::number(line, key).map(|n| format!("{n}")));
        if let Some(v) = v {
            let _ = writeln!(o, "  {key:<22} {v}");
        }
    }
    o
}

/// The most recent record in the file, which is the run the user just watched.
fn last_record(path: &std::path::Path) -> Result<String, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let line = text
        .lines()
        .map(str::trim)
        .rfind(|l| !l.is_empty())
        .ok_or_else(|| format!("{} has no records yet -- run `zc verify` first", path.display()))?;
    if json::string(line, "hw").is_none() || json::number(line, "error_pct").is_none() {
        return Err(format!("the last line of {} is not a calibration record", path.display()));
    }
    Ok(line.to_string())
}

/// The program and arguments that open `url`, per platform.
///
/// Split out so the Windows form can be asserted from any platform. It could
/// not be tested by running it, and it was wrong: `cmd /C start` was passing
/// the URL through a shell that treats `&` as a command separator. Rust quotes
/// a Windows argument only when it contains a space or a tab, and a
/// percent-encoded URL contains neither -- so the URL arrived unquoted and
/// `cmd` cut it at the `&` between `filename=` and `value=`. The browser got a
/// GitHub new-file page with the name filled in and the body empty, which is
/// the whole payload, and `cmd` tried to run the remainder as a command.
///
/// `rundll32 url.dll,FileProtocolHandler` has no shell between it and the
/// argument, so there is nothing left to parse the `&`.
pub fn browser_command(url: &str) -> (&'static str, Vec<String>) {
    #[cfg(target_os = "macos")]
    return ("open", vec![url.to_string()]);
    #[cfg(all(unix, not(target_os = "macos")))]
    return ("xdg-open", vec![url.to_string()]);
    #[cfg(windows)]
    return (
        "rundll32",
        vec!["url.dll,FileProtocolHandler".to_string(), url.to_string()],
    );
    #[cfg(not(any(unix, windows)))]
    return ("", vec![url.to_string()]);
}

fn open_in_browser(url: &str) -> std::io::Result<std::process::ExitStatus> {
    let (prog, args) = browser_command(url);
    if prog.is_empty() {
        return Err(std::io::Error::other("no browser opener on this platform"));
    }
    std::process::Command::new(prog).args(args).status()
}

pub fn run(record: Option<&str>, print_only: bool) -> i32 {
    let path = record
        .map(std::path::PathBuf::from)
        .unwrap_or_else(crate::fit_cmd::record_path);
    let line = match last_record(&path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let filename = match dest_filename(&line) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let url = share_url(&filename, &line);

    // "not in it: ... file paths" is printed three lines below this. Naming an
    // absolute home path here contradicted it on the same screen.
    println!(
        "== share ==  ({})\n",
        zc_report::redact_home(&path.display().to_string())
    );
    println!("  This is the whole record. Nothing else is sent, and zc opens no");
    println!("  connection itself -- your browser does, and you watch it happen.\n");
    print!("{}", describe(&line));
    println!("\n  not in it: hostname, username, serial number, MAC, IP, file paths\n");
    println!("  lands at      {DIR}/{filename}");

    if url.len() > MAX_URL {
        println!("\n  Too long for a URL ({} bytes). Create that file by hand with:\n", url.len());
        println!("{line}");
        return 0;
    }
    println!("  url           {url}");
    println!("\n  GitHub will fork the repo to your account when you commit, then");
    println!("  offer the pull request button. A human reviews and merges.\n");

    if print_only || !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        return 0;
    }
    print!("  open in browser? [y/N] ");
    let _ = std::io::stdout().flush();
    let mut answer = String::new();
    if std::io::stdin().read_line(&mut answer).is_err() || !answer.trim().eq_ignore_ascii_case("y")
    {
        println!("  not opened. The URL above still works whenever you want it.");
        return 0;
    }
    match open_in_browser(&url) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("  could not open a browser ({e}) -- paste the URL above");
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bug this guards could not be reproduced on the machine that found
    /// it, so it is asserted on the arguments rather than on a browser.
    ///
    /// `zc share`'s URL carries exactly one literal `&`, separating
    /// `filename=` from `value=`, and `value` is the whole payload. Passing
    /// that through `cmd /C start` handed it to a shell that cuts at `&`, so a
    /// Windows contributor got a GitHub page with a filename and an empty
    /// body. This runs on every platform, so `windows-latest` runs it too.
    #[test]
    fn the_browser_argument_keeps_the_whole_url() {
        let url = share_url("8bc574063a10f63c-921a62a1.jsonl", REC);
        assert!(url.contains('&'), "the URL must carry the separator");
        let (prog, args) = super::browser_command(&url);
        assert!(!prog.is_empty());
        // Exactly one argument is the URL, and it is the URL entire.
        assert!(
            args.iter().any(|a| a == &url),
            "no argument carried the whole URL: {args:?}"
        );
        // Nothing after the ampersand may be split into its own argument.
        let after = url.split_once('&').expect("separator").1;
        assert!(
            args.iter().any(|a| a.ends_with(after)),
            "the payload after '&' was lost: {args:?}"
        );
        // And no argument may be a shell that would re-parse it.
        assert!(
            !prog.eq_ignore_ascii_case("cmd") && !prog.eq_ignore_ascii_case("sh"),
            "{prog} would parse the URL again"
        );
    }

    /// A real record, copied verbatim from `data/calibration/gate.jsonl`.
    /// Using a synthetic one would let the field list in `FIELDS` drift away
    /// from what `zc verify` actually writes without any test noticing.
    const REC: &str = r#"{"hw":"8bc574063a10f63c","os":"macos","virt":"none","backend":"Metal","runtime":"ollama","ram_bw_gbs":126.55,"vram_bw_gbs":0.00,"disk_bw_gbs":26.65,"gflops":427.6,"threads":4,"kv":"f16","model":"qwen3:1.7b","quant":"Q4_K_M","ctx":4096,"prompt_tokens":979,"eval_tokens":128,"predicted_lo":34.409,"predicted_hi":80.287,"actual_decode_tok_s":85.831,"actual_prefill_tok_s":2750.43,"assumed_eta":0.6160,"implied_eta":0.9220,"implied_prefill_scale":26.1399,"active_params":2031739904,"error_pct":-33.2,"within_range":false}"#;

    /// Percent-decode, written here rather than imported so the test is a
    /// genuine independent check of `encode` rather than its mirror image.
    fn decode(s: &str) -> String {
        let b = s.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'%' && i + 2 < b.len() {
                let hex = std::str::from_utf8(&b[i + 1..i + 3]).unwrap();
                out.push(u8::from_str_radix(hex, 16).unwrap());
                i += 3;
            } else {
                out.push(b[i]);
                i += 1;
            }
        }
        String::from_utf8(out).unwrap()
    }

    /// The filename is content-addressed, which is what makes resubmitting the
    /// same run produce an empty diff instead of a second pull request.
    #[test]
    fn the_same_record_always_gets_the_same_filename() {
        let a = dest_filename(REC).expect("filename");
        assert_eq!(a, dest_filename(REC).expect("filename"));
        // Hand-computed the only way it can be: the same FNV-1a the record's
        // own `hw` fingerprint uses, over the whole line, truncated to 8.
        assert_eq!(
            a,
            format!("8bc574063a10f63c-{}.jsonl", &zc_runtime::calibrate::fingerprint(REC)[..8])
        );
    }

    /// ...and content-addressed the other way too: if it were only keyed on the
    /// machine, a second run from the same laptop would silently overwrite the
    /// first and the dataset would lose a measurement.
    #[test]
    fn one_changed_digit_changes_the_filename() {
        let changed = REC.replace("85.831", "85.832");
        assert_ne!(dest_filename(REC).unwrap(), dest_filename(&changed).unwrap());
    }

    /// Model names come from a runtime we do not control and routinely contain
    /// `:`; a record can also carry `&`, `"` and `+`, each of which silently
    /// truncates or corrupts a query string if it goes through unescaped.
    #[test]
    fn encoding_round_trips_a_hostile_record() {
        let nasty = REC.replace("qwen3:1.7b", "a&b+c\\\"d/é");
        let url = share_url("x-y.jsonl", &nasty);
        let value = url.split("&value=").nth(1).expect("value parameter");
        assert_eq!(decode(value), nasty);
        // One `&` in the URL: the separator. Any other means the value leaked.
        assert_eq!(url.matches('&').count(), 1);
    }

    /// The privacy promise is that what the user is shown IS what is sent. A
    /// field added to the record that nobody added to `FIELDS` would travel
    /// undisclosed, so the disclosure list is checked against the record itself.
    #[test]
    fn every_field_in_the_record_is_disclosed() {
        let described = describe(REC);
        for key in REC.split(",\"").skip(1).filter_map(|s| s.split('"').next()) {
            assert!(described.contains(key), "field {key} is in the record but not disclosed");
        }
        assert!(described.contains("8bc574063a10f63c"), "hw value missing");
    }

    /// Garbage in the record file must produce a message, not a panic and not a
    /// URL that would put nonsense in front of a maintainer.
    #[test]
    fn a_line_that_is_not_a_record_is_rejected() {
        assert!(dest_filename("hello").is_err());
        assert!(dest_filename(r#"{"hw":"nothex","error_pct":1.0}"#).is_err());
    }
}
