# Share Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let anyone who ran `zc verify` contribute their measurement to the dataset in two clicks, without `zc` ever opening a network connection.

**Architecture:** `zc share` prints the record it is about to disclose, derives a content-addressed filename from it, and builds a GitHub web-editor URL with the record prefilled; the browser carries it, GitHub forks on commit, a Python validator gates the PR, and a human merges. Merged records land in a second provenance tier that feeds coefficients but not the headline accuracy claim.

**Tech Stack:** Rust 2024 edition, workspace of 6 crates, zero non-target-gated dependencies. Python 3 standard library only. GitHub Actions with no third-party actions.

**Spec:** `docs/superpowers/specs/2026-08-19-share-loop-design.md`

## Status (2026-08-19) — all six tasks complete

Executed on branch `share-loop`, one commit per task, `./check.sh` green at
every commit. Three deviations, all recorded in the commits:

1. `mod share;` was declared in Task 1 Step 1 rather than Step 5 — the test
   module cannot compile until the module exists, so the "watch it fail" step
   would otherwise have failed for the wrong reason.
2. Task 2 Step 9 exposed a defect the plan did not anticipate: `Fit::merge`
   deduped identical record lines but `Gate` did not, so a community file
   holding a record already in `gate.jsonl` was invisible to the coefficients
   while counting **twice** toward the published accuracy number. Fixed in the
   shared reader (`fit_cmd::read_text`) with its own test, rather than in one
   caller.
3. The validator gained a cross-tier duplicate check off the back of (2). The
   readers now dedupe so it moves no number, but merging one would tell a
   contributor their run landed when nothing was added.

## Global Constraints

- **A number is measured, derived from measured inputs, or printed as `-`.** No fallback constants, ever.
- `zc` makes **zero outbound network connections** in every subcommand, `share` included. The browser connects; `zc` prints a URL.
- `zc-probe` / `zc-bench` / `zc-model` stay dependency-free. No dev-dependencies either; tests use `std` only.
- Python scripts use the standard library only, matching `scripts/ingest_hf.py`.
- `./check.sh` must be green before every commit.
- Repository is `DEEPESH-845/ZeroCloud`, default branch `main`. Binary name is `zc`.
- Test style: assert a hand-computed value or lock in a specific bug, with the reasoning in the doc comment. No test frameworks.
- Coefficients move by `zc fit` from records, never by hand.

---

## File Structure

| File | Responsibility |
|---|---|
| `crates/zc-cli/src/share.rs` (create) | The whole `zc share` surface: read the last record, disclose it, name it, encode it, open it. |
| `crates/zc-cli/src/main.rs` (modify) | `mod share;`, `--record`/`--print` flags, dispatch before the hardware probe, help text. |
| `crates/zc-cli/src/fit_cmd.rs` (modify) | `sources()` and `read_text()` — the dataset is now a list of files, not one file. |
| `crates/zc-model/src/fit.rs` (modify) | `Fit::from_text()` so a caller can supply concatenated text; `EMBEDDED` becomes build-script generated. |
| `crates/zc-cli/src/gate_cmd.rs` (modify) | Compute the gate twice: curated for the headline and the exit code, curated+community for the second line. |
| `crates/zc-cli/src/doctor.rs` (modify) | One call site follows `summary_text`'s new signature. |
| `crates/zc-model/build.rs` (modify) | Embed the curated file plus every `community/*.jsonl`. |
| `scripts/validate_calibration.py` (create) | Repo-wide dataset validation, with `--self-test`. |
| `.github/workflows/calibration-prs.yml` (create) | Run the validator on any PR touching `data/calibration/**`. |
| `.github/workflows/ci.yml` (modify) | Run the validator on `main` too. |
| `check.sh` (modify) | Run the validator locally. |
| `data/calibration/community/README.md` (create) | Why the directory exists and what a file in it means. |
| `README.md`, `CONTRIBUTING.md`, `docs/gate-runbook.md` (modify) | Document the loop end to end. |

---

## Task 1: `zc share`

Implements spec §1 and §3. Self-contained: no other task depends on it compiling, and it depends on nothing but what already exists.

**Files:**
- Create: `crates/zc-cli/src/share.rs`
- Modify: `crates/zc-cli/src/main.rs:1-6` (module list), `:97-120` (flag stripping), `:146-151` (dispatch), `:11-20` (help)
- Test: `crates/zc-cli/src/share.rs` (a `#[cfg(test)] mod tests` at the end of the file)

**Interfaces:**
- Consumes: `zc_model::json::{string, number, boolean}`; `zc_runtime::calibrate::fingerprint(&str) -> String` (FNV-1a-64, 16 lowercase hex); `crate::fit_cmd::record_path() -> PathBuf`.
- Produces: `share::run(record: Option<&str>, print_only: bool) -> i32`, plus `share::dest_filename(&str) -> Result<String, String>`, `share::share_url(&str, &str) -> String` and `share::encode(&str) -> String` for the tests.

- [x] **Step 1: Write the failing tests**

Create `crates/zc-cli/src/share.rs` containing *only* this test module for now:

```rust
#[cfg(test)]
mod tests {
    use super::*;

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
        let b = dest_filename(&REC.to_string()).expect("filename");
        assert_eq!(a, b);
        assert_eq!(a, format!("8bc574063a10f63c-{}.jsonl", &zc_runtime::calibrate::fingerprint(REC)[..8]));
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
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p zc-cli share 2>&1 | tail -20`
Expected: FAIL — `cannot find function 'dest_filename'`, `share_url`, `describe`.

- [x] **Step 3: Write the implementation**

Prepend to `crates/zc-cli/src/share.rs`, above the test module:

```rust
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
const DIR: &str = "data/calibration/community";

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
    "hw", "os", "virt", "backend", "runtime", "ram_bw_gbs", "vram_bw_gbs",
    "disk_bw_gbs", "gflops", "threads", "kv", "model", "quant", "ctx",
    "prompt_tokens", "eval_tokens", "predicted_lo", "predicted_hi",
    "actual_decode_tok_s", "actual_prefill_tok_s", "assumed_eta", "implied_eta",
    "implied_prefill_scale", "active_params", "error_pct", "within_range",
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
        .filter(|l| !l.is_empty())
        .next_back()
        .ok_or_else(|| format!("{} has no records yet -- run `zc verify` first", path.display()))?;
    if json::string(line, "hw").is_none() || json::number(line, "error_pct").is_none() {
        return Err(format!("the last line of {} is not a calibration record", path.display()));
    }
    Ok(line.to_string())
}

fn open_in_browser(url: &str) -> std::io::Result<std::process::ExitStatus> {
    #[cfg(target_os = "macos")]
    return std::process::Command::new("open").arg(url).status();
    #[cfg(all(unix, not(target_os = "macos")))]
    return std::process::Command::new("xdg-open").arg(url).status();
    #[cfg(windows)]
    return std::process::Command::new("cmd").args(["/C", "start", ""]).arg(url).status();
    #[cfg(not(any(unix, windows)))]
    return Err(std::io::Error::other("no browser opener on this platform"));
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

    println!("== share ==  ({})\n", path.display());
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
    if std::io::stdin().read_line(&mut answer).is_err() || !answer.trim().eq_ignore_ascii_case("y") {
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
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p zc-cli share 2>&1 | tail -20`
Expected: 5 passed.

- [x] **Step 5: Wire it into the CLI**

In `crates/zc-cli/src/main.rs`, add `mod share;` to the module list (alphabetical, after `mod machine;`).

Add the two flags immediately after the `--top` block, before `let cmd = ...`:

```rust
    // `share` reads a record file rather than measuring, so both of its flags
    // are stripped here with the other globals and never reach the probe.
    let record = take_value(&mut args, "--record");
    let print_only = take_flag(&mut args, "--print");
```

Add the dispatch beside `fit` and `gate`, which is what keeps `zc share` from running a 20-second benchmark it has no use for:

```rust
    if cmd == "share" {
        std::process::exit(share::run(record.as_deref(), print_only));
    }
```

Add to `HELP`, after the `zc gate` line:

```
    zc share [--record FILE] [--print]
                          submit your last `zc verify` measurement upstream
```

- [x] **Step 6: Prove it end to end against a real record**

```bash
cargo build --release --bin zc
./target/release/zc share --print | tail -20
./target/release/zc share --record /nonexistent; echo "exit: $?"
```

Expected: the first prints the record, the destination filename and a URL beginning `https://github.com/DEEPESH-845/ZeroCloud/new/main?filename=`. The second prints `cannot read /nonexistent: ...` and `exit: 1`.

- [x] **Step 7: Full check and commit**

```bash
./check.sh
git add crates/zc-cli/src/share.rs crates/zc-cli/src/main.rs
git commit -m "feat: zc share, without a socket

The record travels in a URL the browser opens, so zc still makes no
outbound connection and there is no token to store. GitHub forks on
commit for anyone without write access, so the fork-and-PR half needs
no code. The filename is content-addressed, which makes resubmitting
the same run an empty diff rather than a second pull request.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Two provenance tiers

Implements spec §2. The dataset stops being one file.

**Files:**
- Modify: `crates/zc-model/src/fit.rs:365-368` (`load`), add `from_text`
- Modify: `crates/zc-cli/src/fit_cmd.rs:8-10` (constants), `:53-75` (`resolve`, `run`, `summary_text`)
- Modify: `crates/zc-cli/src/gate_cmd.rs:16` and `:76`
- Modify: `crates/zc-cli/src/doctor.rs:22`
- Modify: `crates/zc-cli/src/main.rs:172`
- Create: `data/calibration/community/README.md`
- Test: `crates/zc-cli/src/fit_cmd.rs` (existing `mod tests`)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `fit_cmd::sources() -> Vec<PathBuf>` (curated first, then sorted `community/*.jsonl`), `fit_cmd::read_text(&[PathBuf]) -> String`, `fit_cmd::summary_text(&[PathBuf]) -> String`, `zc_model::Fit::from_text(&str) -> Fit`.

- [x] **Step 1: Write the failing tests**

Append to the existing `mod tests` in `crates/zc-cli/src/fit_cmd.rs`:

```rust
    /// A merged submission has to be read without anyone editing a path list,
    /// or the contribution loop needs a maintainer in it twice.
    #[test]
    fn community_records_join_the_curated_file() {
        let d = scratch("community");
        std::fs::write(d.join("gate.jsonl"), "").unwrap();
        std::fs::create_dir_all(d.join("community")).unwrap();
        std::fs::write(d.join("community").join("b-2.jsonl"), "").unwrap();
        std::fs::write(d.join("community").join("a-1.jsonl"), "").unwrap();
        // Not a record file, and must not be read as one.
        std::fs::write(d.join("community").join("README.md"), "").unwrap();

        let got = super::sources_in(&d);
        assert_eq!(
            got,
            vec![
                d.join("gate.jsonl"),
                d.join("community").join("a-1.jsonl"),
                d.join("community").join("b-2.jsonl"),
            ],
            "curated first, then community sorted, and only .jsonl"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The common case is a user with no community directory at all. It must
    /// behave exactly as it did before the directory existed.
    #[test]
    fn a_missing_community_directory_is_not_an_error() {
        let d = scratch("nocommunity");
        std::fs::write(d.join("gate.jsonl"), "").unwrap();
        assert_eq!(super::sources_in(&d), vec![d.join("gate.jsonl")]);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Concatenation has to normalise line endings: the web editor commits
    /// whatever the contributor's browser sent, and a file with no trailing
    /// newline would otherwise weld two records into one unparseable line.
    #[test]
    fn concatenation_separates_records_that_lack_a_trailing_newline() {
        let d = scratch("concat");
        std::fs::write(d.join("gate.jsonl"), "{\"a\":1}").unwrap();
        std::fs::create_dir_all(d.join("community")).unwrap();
        std::fs::write(d.join("community").join("x-1.jsonl"), "{\"b\":2}\n").unwrap();
        let text = super::read_text(&super::sources_in(&d));
        assert_eq!(text, "{\"a\":1}\n{\"b\":2}\n");
        let _ = std::fs::remove_dir_all(&d);
    }
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p zc-cli fit_cmd 2>&1 | tail -20`
Expected: FAIL — `cannot find function 'sources_in'` and `read_text`.

- [x] **Step 3: Implement the source list**

In `crates/zc-cli/src/fit_cmd.rs`, add beside the existing constants:

```rust
const COMMUNITY: &str = "community";
```

Add after `resolve`:

```rust
/// Every file the dataset is read from: the curated set, then each merged
/// community submission, sorted so the concatenation is byte-stable across
/// machines and two readers cannot compute different numbers from one repo.
pub fn sources() -> Vec<std::path::PathBuf> {
    match env_override() {
        Some(p) => vec![p],
        None => sources_in(std::path::Path::new(DEFAULT_DIR)),
    }
}

fn sources_in(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = vec![resolve(dir)];
    let mut community: Vec<std::path::PathBuf> = std::fs::read_dir(dir.join(COMMUNITY))
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    community.sort();
    out.extend(community);
    out
}

/// Concatenate the sources, one record per line.
///
/// Trimming and re-terminating every line is load-bearing: a community file
/// arrives from GitHub's web editor with whatever trailing newline the
/// contributor's browser decided on, and a missing one would weld the next
/// record onto the end of it.
pub fn read_text(paths: &[std::path::PathBuf]) -> String {
    let mut out = String::new();
    for p in paths {
        let Ok(text) = std::fs::read_to_string(p) else { continue };
        for line in text.lines() {
            let line = line.trim();
            if !line.is_empty() {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out
}
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p zc-cli fit_cmd 2>&1 | tail -20`
Expected: 6 passed (3 existing plus 3 new).

- [x] **Step 5: Let `Fit` take text rather than a path**

In `crates/zc-model/src/fit.rs`, replace `load`:

```rust
    /// Fit from already-concatenated records, so a caller assembling several
    /// files does not have to make `fit` care how many there are.
    pub fn from_text(local: &str) -> Self {
        Self::from_records(&parse_records(&merge(EMBEDDED, local)))
    }

    pub fn load(path: &std::path::Path) -> Self {
        Self::from_text(&std::fs::read_to_string(path).unwrap_or_default())
    }
```

- [x] **Step 6: Point every reader at the source list**

`crates/zc-cli/src/fit_cmd.rs` — `run` and `summary_text`:

```rust
pub fn run() -> i32 {
    print!("{}", summary_text(&sources()));
    0
}

/// The body of `zc fit`, as text.
///
/// Returned rather than printed so `zc doctor` can embed the identical block.
/// A bug report that shows different coefficients from the ones the user sees
/// would send everyone chasing a discrepancy that is not there.
pub fn summary_text(paths: &[std::path::PathBuf]) -> String {
    use std::fmt::Write;
    let fit = Fit::from_text(&read_text(paths));
    let mut o = String::new();
```

and, in the same function, replace both existing uses of `path.display()` in
its header lines with `describe_sources(paths)`, adding:

```rust
/// What `zc fit` and `zc gate` say they read. Naming the community count
/// separately is the point: it is how a contributor sees their record arrive.
fn describe_sources(paths: &[std::path::PathBuf]) -> String {
    match paths.len() {
        0 => "no dataset".to_string(),
        1 => paths[0].display().to_string(),
        n => format!("{}, plus {} community record(s)", paths[0].display(), n - 1),
    }
}
```

`crates/zc-cli/src/doctor.rs:22`:

```rust
        zc_report::markdown::render(&report, &crate::fit_cmd::summary_text(&crate::fit_cmd::sources()))
```

`crates/zc-cli/src/main.rs:172`:

```rust
    let fit = zc_model::Fit::from_text(&fit_cmd::read_text(&fit_cmd::sources()));
```

- [x] **Step 7: Make `zc gate` report both tiers**

In `crates/zc-cli/src/gate_cmd.rs`, replace the whole opening of `run` — from
`let path = ...` down to and including the closing brace of the `let ... else`
block — with this. The `let Ok(text) = read_to_string` form has to go: with two
tiers, "no data" means no *records*, not a missing file, and a repository that
has a `community/` directory but an absent `gate.jsonl` would otherwise take
the early exit while holding perfectly good records.

```rust
    let curated = crate::fit_cmd::path();
    let all = crate::fit_cmd::sources();
    let text = crate::fit_cmd::read_text(std::slice::from_ref(&curated));
    if text.trim().is_empty() {
        println!("No calibration data at {}.\n", curated.display());
        println!("The Phase 0 gate is: median decode error < {MAX_MEDIAN_ERROR_PCT:.0}% across >= {MIN_MACHINES} machines.");
        println!("Nothing has been measured, so it is unmeasurable — not failed, unknown.\n");
        println!("    ollama pull qwen3:4b && zc verify");
        return FAILED;
    }
```

Then change the header line below it from `path.display()` to
`curated.display()`.

After the `within_range` line, add the second tier:

```rust
    // The headline number and the exit code come from the curated tier alone.
    // Community records are real evidence and they move every coefficient, but
    // a claim about our own accuracy must rest on records whose provenance we
    // can state. Printing both, always, is what stops that from reading as
    // cherry-picking.
    if all.len() > 1 {
        let g_all = Gate::from_records(&parse_records(&crate::fit_cmd::read_text(&all)));
        println!(
            "\n  with {} community record(s):   {:.1}%   over {} machine(s)",
            all.len() - 1,
            g_all.median_pct,
            g_all.machines.len()
        );
    }
```

and change `let fit = Fit::load(&path);` to:

```rust
    let fit = Fit::from_text(&crate::fit_cmd::read_text(&all));
```

- [x] **Step 8: Add the directory and its README**

```bash
mkdir -p data/calibration/community
```

Create `data/calibration/community/README.md`:

```markdown
# Community calibration records

One file per submitted `zc verify` run, named `<hw>-<hash8>.jsonl`, holding
exactly one record on one line. `zc share` produces both the name and the
content; `scripts/validate_calibration.py` checks every file here on every
pull request.

These records feed `zc fit`, so a merged submission improves the coefficients
behind everyone's predictions. They do **not** move the headline accuracy
figure `zc gate` prints, which is computed from `../gate.jsonl` — the tier
whose provenance we can state. `zc gate` prints both numbers.

Promotion from here into `gate.jsonl` is a deliberate `git mv` with a human's
name on the commit. There is no tooling for it on purpose.
```

- [x] **Step 9: Prove the two tiers behave**

```bash
cargo build --release --bin zc
./target/release/zc gate | tail -6           # unchanged: no community records yet
cp data/calibration/gate.jsonl /tmp/one.jsonl && head -1 /tmp/one.jsonl > data/calibration/community/test-1.jsonl
./target/release/zc fit | head -3            # header names 1 community record
./target/release/zc gate | grep community    # the second tier line appears
rm data/calibration/community/test-1.jsonl
```

Expected: the `fit` header reads `..., plus 1 community record(s)`, and `gate`
prints a `with 1 community record(s)` line. Both disappear when the file is
removed.

- [x] **Step 10: Full check and commit**

```bash
./check.sh
git add crates data/calibration/community
git commit -m "feat: two provenance tiers for the dataset

Community submissions feed every coefficient the day they merge, so a
contributor's run improves everyone's predictions. The headline
accuracy number and the exit code stay on the curated tier, whose
provenance we can state. Both numbers print, always.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: The validator

Implements spec §4. Standalone: it reads the repository and exits non-zero, and
nothing in Rust depends on it.

**Files:**
- Create: `scripts/validate_calibration.py`
- Test: the same file, under `--self-test`

**Interfaces:**
- Consumes: `data/calibration/gate.jsonl` and `data/calibration/community/*.jsonl` as they exist on disk.
- Produces: exit code 0 (valid) or 1 (any failure), with one line per failure on stdout. `--self-test` runs the fixtures and touches no repository file.

- [x] **Step 1: Write the script, tests first inside it**

Create `scripts/validate_calibration.py`:

```python
#!/usr/bin/env python3
"""Validate every calibration record in the repository.

Runs over the whole tree rather than only the files a pull request touched:
repo-wide integrity is the invariant, and a change that corrupts an untouched
file is precisely what a changed-files-only validator misses.

The important checks are not schema checks. A record carries the inputs its
own conclusions were computed from, so `error_pct` and `within_range` are both
recomputable here -- which means a fabricated "0% error" submission has to
contradict itself to exist. It cannot stop a consistent lie from a machine
nobody can inspect; that is what the two provenance tiers and a human on the
merge button are for.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "data" / "calibration"
NAME_RE = re.compile(r"^[0-9a-f]{16}-[0-9a-f]{8}\.jsonl$")
MAX_BYTES = 4096
VIRT = {"none", "hypervisor", "wsl2", "container"}

REQUIRED = [
    "hw", "os", "virt", "backend", "runtime", "ram_bw_gbs", "vram_bw_gbs",
    "disk_bw_gbs", "gflops", "threads", "kv", "model", "quant", "ctx",
    "prompt_tokens", "eval_tokens", "predicted_lo", "predicted_hi",
    "actual_decode_tok_s", "actual_prefill_tok_s", "assumed_eta",
    "implied_eta", "implied_prefill_scale", "active_params", "error_pct",
    "within_range",
]
ALLOWED = set(REQUIRED)

# Bounds a schema cannot express. Wide on purpose: the job is to reject the
# physically impossible, not to have an opinion about what hardware exists.
RANGES = {
    "ram_bw_gbs": (1.0, 2000.0),
    "gflops": (0.0, 1e6),
    "threads": (1, 1024),
    "ctx": (1, 10_000_000),
    "implied_eta": (0.0, 1.5),
}


def fnv1a(s):
    """FNV-1a-64, byte-identical to `zc_runtime::calibrate::fingerprint`."""
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def check_record(rec, where, out):
    missing = [k for k in REQUIRED if k not in rec]
    for key in missing:
        out.append(f"{where}: missing required field {key!r}")
    # Every check below indexes fields directly, so a record missing any of
    # them is reported and abandoned rather than crashing the whole run.
    if missing:
        return
    extra = set(rec) - ALLOWED
    if extra:
        out.append(f"{where}: unknown field(s) {sorted(extra)}")
    if rec["virt"] not in VIRT:
        out.append(f"{where}: virt {rec['virt']!r} not one of {sorted(VIRT)}")
    for key, (lo, hi) in RANGES.items():
        v = rec[key]
        if not isinstance(v, (int, float)) or not (lo < v <= hi):
            out.append(f"{where}: {key}={v} outside ({lo}, {hi}]")
    lo, hi, act = rec["predicted_lo"], rec["predicted_hi"], rec["actual_decode_tok_s"]
    if not (0 < lo <= hi):
        out.append(f"{where}: predicted range {lo}-{hi} is not ordered and positive")
    if act <= 0:
        out.append(f"{where}: actual_decode_tok_s={act} is not positive")
        return
    # The two claims the record makes about itself, recomputed from the inputs
    # it carries. `zc verify` rounds error to one decimal place, hence 0.15.
    mid = (lo + hi) / 2
    want_err = (mid - act) / act * 100
    if abs(want_err - rec["error_pct"]) > 0.15:
        out.append(
            f"{where}: error_pct={rec['error_pct']} but its own numbers give {want_err:.1f}"
        )
    want_in = lo <= act <= hi
    if want_in != rec["within_range"]:
        out.append(
            f"{where}: within_range={rec['within_range']} but {act} vs {lo}-{hi} gives {want_in}"
        )


def check_community_file(path, out):
    where = path.name
    raw = path.read_bytes()
    if len(raw) > MAX_BYTES:
        out.append(f"{where}: {len(raw)} bytes, over the {MAX_BYTES} limit")
        return
    if not NAME_RE.match(where):
        out.append(f"{where}: filename must be <16-hex>-<8-hex>.jsonl")
        return
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        out.append(f"{where}: not valid UTF-8 ({e})")
        return
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) != 1:
        out.append(f"{where}: holds {len(lines)} records, must hold exactly 1")
        return
    line = lines[0].strip()
    try:
        rec = json.loads(line)
    except json.JSONDecodeError as e:
        out.append(f"{where}: not valid JSON ({e})")
        return
    if not isinstance(rec, dict):
        out.append(f"{where}: top level is {type(rec).__name__}, must be an object")
        return
    hw, digest = where[:-len(".jsonl")].split("-")
    if rec.get("hw") != hw:
        out.append(f"{where}: filename says hw={hw}, record says {rec.get('hw')!r}")
    if fnv1a(line)[:8] != digest:
        out.append(f"{where}: filename hash {digest} does not match its contents")
    check_record(rec, where, out)


def validate():
    out = []
    curated = CALIB / "gate.jsonl"
    if curated.is_file():
        for i, line in enumerate(curated.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                check_record(json.loads(line), f"gate.jsonl:{i}", out)
            except json.JSONDecodeError as e:
                out.append(f"gate.jsonl:{i}: not valid JSON ({e})")
    community = CALIB / "community"
    if community.is_dir():
        for p in sorted(community.iterdir()):
            if p.suffix == ".jsonl":
                check_community_file(p, out)
    return out


GOOD = {
    "hw": "8bc574063a10f63c", "os": "macos", "virt": "none", "backend": "Metal",
    "runtime": "ollama", "ram_bw_gbs": 126.55, "vram_bw_gbs": 0.0,
    "disk_bw_gbs": 5.01, "gflops": 427.6, "threads": 4, "kv": "f16",
    "model": "qwen3:1.7b", "quant": "Q4_K_M", "ctx": 4096,
    "prompt_tokens": 979, "eval_tokens": 128, "predicted_lo": 60.0,
    "predicted_hi": 100.0, "actual_decode_tok_s": 80.0,
    "actual_prefill_tok_s": 2750.43, "assumed_eta": 0.616,
    "implied_eta": 0.922, "implied_prefill_scale": 26.1399,
    "active_params": 2031739904, "error_pct": 0.0, "within_range": True,
}


def self_test():
    """Hand-computed: midpoint of 60-100 is 80, actual is 80, so error is
    exactly 0.0% and the measurement is inside the range. Every fixture below
    perturbs one field of that and must be caught."""
    failures = []

    def case(name, rec, should_pass):
        out = []
        check_record(dict(rec), "fixture", out)
        ok = not out
        if ok != should_pass:
            failures.append(f"{name}: expected {'pass' if should_pass else 'fail'}, got {out}")

    case("a correct record", GOOD, True)
    case("error_pct contradicts its own inputs", {**GOOD, "error_pct": 0.0, "actual_decode_tok_s": 40.0}, False)
    case("within_range contradicts its own bounds", {**GOOD, "within_range": False}, False)
    case("impossible eta", {**GOOD, "implied_eta": 2.0}, False)
    case("unknown virt", {**GOOD, "virt": "xen"}, False)
    case("smuggled field", {**GOOD, "note": "trust me"}, False)
    case("missing error_pct", {k: v for k, v in GOOD.items() if k != "error_pct"}, False)

    # Filename-level fixtures need a real file, written outside the repo tree.
    import tempfile
    line = json.dumps(GOOD, separators=(",", ":"))
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        good_name = f"{GOOD['hw']}-{fnv1a(line)[:8]}.jsonl"
        for name, content, should_pass in [
            (good_name, line + "\n", True),
            (good_name, line + "\n" + line + "\n", False),
            (f"{GOOD['hw']}-deadbeef.jsonl", line + "\n", False),
            ("not-a-record.jsonl", line + "\n", False),
        ]:
            p = d / name
            p.write_text(content)
            out = []
            check_community_file(p, out)
            if (not out) != should_pass:
                failures.append(f"{name} ({len(content)}B): expected {'pass' if should_pass else 'fail'}, got {out}")
            p.unlink()

    for f in failures:
        print(f"SELF-TEST FAIL  {f}")
    print(f"self-test: {'ok' if not failures else str(len(failures)) + ' failure(s)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    problems = validate()
    for p in problems:
        print(f"INVALID  {p}")
    print(f"{'FAILED' if problems else 'ok'}: {len(problems)} problem(s) in {CALIB}")
    sys.exit(1 if problems else 0)
```

- [x] **Step 2: Run the self-test**

Run: `python3 scripts/validate_calibration.py --self-test`
Expected: `self-test: ok`, exit 0. If any fixture reports the wrong verdict, fix
the check it exercises — not the fixture.

- [x] **Step 3: Run it against the real dataset**

Run: `python3 scripts/validate_calibration.py`
Expected: `ok: 0 problem(s)`. If `gate.jsonl` reports problems, **do not edit the
data to make them go away** — bring the finding back; a real record failing a
plausibility bound is either a validator bug or a genuine discovery.

- [x] **Step 4: Prove it rejects a hand-edited record**

```bash
cp data/calibration/gate.jsonl /tmp/gate.bak
python3 - <<'EOF'
import json
p = "data/calibration/gate.jsonl"
lines = open(p).read().splitlines()
rec = json.loads(lines[0]); rec["error_pct"] = 0.0
lines[0] = json.dumps(rec, separators=(",", ":"))
open(p, "w").write("\n".join(lines) + "\n")
EOF
python3 scripts/validate_calibration.py; echo "exit: $?"
cp /tmp/gate.bak data/calibration/gate.jsonl
python3 scripts/validate_calibration.py; echo "exit: $?"
```

Expected: `exit: 1` with an `error_pct=0.0 but its own numbers give ...` line,
then `exit: 0` after the restore. Confirm `git diff --stat data/` is empty.

- [x] **Step 5: Commit**

```bash
git add scripts/validate_calibration.py
git commit -m "feat: validate every calibration record, repo-wide

A record carries the inputs its own conclusions came from, so error_pct
and within_range are both recomputable here. A fabricated submission
has to contradict itself to exist. Runs over the whole tree, not just
changed files: a PR that corrupts an untouched record is exactly what
a changed-files-only validator misses.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: CI

Implements spec §5.

**Files:**
- Create: `.github/workflows/calibration-prs.yml`
- Modify: `.github/workflows/ci.yml` (the `data` job)
- Modify: `check.sh` (before the `installer` section)

**Interfaces:**
- Consumes: `scripts/validate_calibration.py` from Task 3.
- Produces: nothing consumed by later tasks.

- [x] **Step 1: Write the pull-request workflow**

Create `.github/workflows/calibration-prs.yml`:

```yaml
# Validate calibration submissions on every pull request that touches them.
#
# This is the *only* automated gate on a community record, by design. There is
# no auto-merge and no bot approval: a check status tells a human whether the
# file is well-formed and internally consistent, and a human decides whether to
# trust the machine it came from. Automating the second half would mean
# publishing an accuracy claim nobody looked at.
name: calibration-prs

on:
  pull_request:
    paths:
      - "data/calibration/**"
      - "scripts/validate_calibration.py"

permissions:
  contents: read

jobs:
  validate:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v4
      # Self-test first: a validator whose own fixtures fail cannot be trusted
      # to have judged the submission, and it would fail open otherwise.
      - run: python3 scripts/validate_calibration.py --self-test
      - run: python3 scripts/validate_calibration.py
```

- [x] **Step 2: Wire it into `ci.yml`**

In `.github/workflows/ci.yml`, in the `data` job, add before the `cargo build` step:

```yaml
      - run: python3 scripts/validate_calibration.py --self-test
      - run: python3 scripts/validate_calibration.py
```

- [x] **Step 3: Wire it into `check.sh`**

In `check.sh`, insert before the `echo "== installer =="` line:

```bash
echo "== calibration =="
python3 scripts/validate_calibration.py --self-test
python3 scripts/validate_calibration.py
```

- [x] **Step 4: Verify both locally**

```bash
./check.sh 2>&1 | grep -A3 "== calibration =="
python3 -c "import yaml,sys" 2>/dev/null && python3 -c "
import yaml
for f in ['.github/workflows/calibration-prs.yml', '.github/workflows/ci.yml']:
    yaml.safe_load(open(f)); print(f, 'parses')
" || echo "pyyaml absent - skipping the syntax parse, CI will catch it"
```

Expected: the calibration section prints `self-test: ok` and `ok: 0 problem(s)`,
and `check.sh` still ends in `OK`.

- [x] **Step 5: Commit**

```bash
git add .github/workflows/calibration-prs.yml .github/workflows/ci.yml check.sh
git commit -m "ci: gate calibration submissions on the validator

Validation is the only automated gate, and a human merges. The
self-test runs first: a validator whose own fixtures fail would
otherwise fail open and wave a bad record through.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Embed both tiers in the binary

Implements spec §7. Without this, an installed user predicts from the curated
tier alone and a merged submission never reaches anyone's binary.

**Files:**
- Modify: `crates/zc-model/build.rs`
- Modify: `crates/zc-model/src/fit.rs:239-241` (`EMBEDDED`), `:455-460` (its test)

**Interfaces:**
- Consumes: `data/calibration/gate.jsonl`, `data/calibration/community/*.jsonl`.
- Produces: generated `pub static EMBEDDED_CALIBRATION: &[&str]`, and `fit::EMBEDDED` becomes `fn embedded() -> String`.

- [x] **Step 1: Write the failing test**

In `crates/zc-model/src/fit.rs`, replace `the_binary_ships_a_non_empty_dataset`:

```rust
    /// The binary must not ship an empty dataset. If the calibration files are
    /// ever emptied or moved, every installed `zc` silently reverts to priors,
    /// and nothing else in the test suite would notice.
    ///
    /// The community tier is embedded too, so a merged submission reaches the
    /// next release's users rather than only the repository.
    #[test]
    fn the_binary_ships_a_non_empty_dataset() {
        let text = super::embedded();
        let n = super::parse_records(&text).len();
        assert!(n >= 5, "embedded dataset has only {n} records");
        // Every embedded chunk is newline-terminated before concatenation, or
        // the last record of one file and the first of the next become one
        // unparseable line and both are silently dropped.
        assert_eq!(text.lines().count(), text.trim_end().lines().count());
        assert!(!text.contains("}{"), "two records were welded together");
    }
```

- [x] **Step 2: Run it to verify it fails**

Run: `cargo test -p zc-model the_binary_ships 2>&1 | tail -10`
Expected: FAIL — `cannot find function 'embedded'`.

- [x] **Step 3: Extend the build script**

In `crates/zc-model/build.rs`, add before the final `std::fs::write`:

```rust
    // The calibration dataset, both tiers. Embedded rather than read at runtime
    // for the same reason as the catalog: an installed user has no repository,
    // and a binary that cannot see the dataset predicts from priors while the
    // README quotes an accuracy figure computed from records it never loaded.
    let calib = PathBuf::from("../../data/calibration");
    println!("cargo:rerun-if-changed={}", calib.display());
    let mut records: Vec<PathBuf> = vec![calib.join("gate.jsonl")];
    let community = calib.join("community");
    println!("cargo:rerun-if-changed={}", community.display());
    let mut merged: Vec<PathBuf> = std::fs::read_dir(&community)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    merged.sort();
    records.extend(merged);

    let mut cal = String::from(
        "/// Calibration records embedded at build time. Generated by build.rs.\n\
         pub static EMBEDDED_CALIBRATION: &[&str] = &[\n",
    );
    for path in &records {
        if !path.is_file() {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let abs = std::fs::canonicalize(path)
            .unwrap_or_else(|e| panic!("cannot resolve {}: {e}", path.display()));
        cal.push_str(&format!("    include_str!(r\"{}\"),\n", abs.display()));
    }
    cal.push_str("];\n");
    out.push_str(&cal);
```

- [x] **Step 4: Replace the hardcoded include in `fit.rs`**

In `crates/zc-model/src/fit.rs`, replace the `EMBEDDED` constant:

```rust
/// Every calibration record compiled into this binary: the curated dataset
/// plus every merged community submission.
///
/// Generated by `build.rs` rather than a hardcoded `include_str!`, because the
/// community tier is a directory whose contents change with every merged pull
/// request and a hand-maintained list would go stale on the first one.
///
/// Each file is trimmed and re-terminated before joining. A file committed
/// through GitHub's web editor arrives with whatever trailing newline the
/// contributor's browser chose, and one missing newline would weld two records
/// into a single unparseable line.
pub fn embedded() -> String {
    let mut out = String::new();
    for chunk in crate::catalog::EMBEDDED_CALIBRATION {
        for line in chunk.lines() {
            let line = line.trim();
            if !line.is_empty() {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out
}
```

Replace its use in `from_text`:

```rust
    pub fn from_text(local: &str) -> Self {
        Self::from_records(&parse_records(&merge(&embedded(), local)))
    }
```

`crates/zc-model/src/catalog.rs:14` does `include!(concat!(env!("OUT_DIR"),
"/catalog_embedded.rs"))`, so everything `build.rs` generates lands directly in
that module — hence `crate::catalog::EMBEDDED_CALIBRATION`, alongside the
catalog's own `EMBEDDED`.

- [x] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p zc-model 2>&1 | tail -6`
Expected: all pass.

- [x] **Step 6: Prove a community record reaches the binary**

```bash
head -1 data/calibration/gate.jsonl > data/calibration/community/aaaaaaaaaaaaaaaa-deadbeef.jsonl
cargo build --release --bin zc && ./target/release/zc fit | head -6
rm data/calibration/community/aaaaaaaaaaaaaaaa-deadbeef.jsonl
cargo build --release --bin zc && ./target/release/zc fit | head -6
```

Expected: the run count for one bucket rises by 1 while the file exists (the
machine count does not — it is the same `hw`), and returns when it is removed.
This is also a live check that the machines-not-runs confidence rule holds.

- [x] **Step 7: Full check and commit**

```bash
./check.sh
git add crates/zc-model/build.rs crates/zc-model/src/fit.rs
git commit -m "feat: embed both calibration tiers at build time

A merged submission has to reach the next release's binaries, not just
the repository, or an installed user predicts from priors while the
README quotes a number computed from records their binary never saw.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Document the loop

The loop is only real if a stranger can find it. Every claim added here must
already be true when the commit lands.

**Files:**
- Modify: `README.md` (the contribution section), `CONTRIBUTING.md`, `docs/gate-runbook.md` (step 6)
- Modify: `crates/zc-cli/src/verify.rs:271` (the message pointing at `zc share`)

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: nothing.

- [x] **Step 1: Capture the real output**

```bash
./target/release/zc share --print > /tmp/zc-share.txt
cat /tmp/zc-share.txt
```

- [x] **Step 2: Add the loop to `README.md`**

Under the existing "Adding a machine is the single most useful contribution
right now" line, add a section pasted from `/tmp/zc-share.txt` — the record
disclosure block and the URL line, abridged with `...` only where a line is
identical to one already shown. State plainly: `zc share` opens no connection,
the browser does; GitHub forks on commit; a validator checks the file and a
human merges; merged records feed the coefficients immediately and the headline
number only after a maintainer promotes them.

- [x] **Step 3: Add the same three steps to `CONTRIBUTING.md`**

`ollama pull qwen3:1.7b` → `zc verify qwen3:1.7b` → `zc share`. Note that
`scripts/validate_calibration.py` runs locally and is the same script CI runs,
so a contributor can check their file before opening the PR.

- [x] **Step 4: Update the runbook**

In `docs/gate-runbook.md`, step 6 currently says to carry `local.jsonl` back by
hand. Add `zc share` as the path for anyone who is not the maintainer, keeping
the manual `cat >> gate.jsonl` route for machines the maintainer runs
themselves — those go straight into the curated tier and that is the whole
distinction.

- [x] **Step 5: Fix the pointer in `zc verify`**

`crates/zc-cli/src/verify.rs:271` already says ``(local only; `zc share` to submit)``.
Verify the command it names now exists and behaves as described:

```bash
./target/release/zc verify --help 2>&1 | head -3 || true
./target/release/zc share --print | head -3
```

- [x] **Step 6: Verify every path named in the docs resolves**

```bash
grep -oE '(docs|scripts|data)/[A-Za-z0-9_./-]+' README.md CONTRIBUTING.md docs/gate-runbook.md \
  | cut -d: -f2 | sort -u | while read -r f; do [ -e "$f" ] || echo "MISSING: $f"; done
```
Expected: no output.

- [x] **Step 7: Full check and commit**

```bash
./check.sh
git add README.md CONTRIBUTING.md docs/gate-runbook.md
git commit -m "docs: the share loop, end to end

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Out of scope

Named so nobody adds them mid-plan:

- GitHub device flow, token caching, forking or opening pull requests from the CLI.
- Web result cards (§2.11), `zc serve` (§2.10), `zc-tui` (§2.9), `zc plan` (§2.5), live HF lookup (§2.14).
- Tooling to promote a record from `community/` into `gate.jsonl`. A `git mv` is the promotion, and it should cost a visible commit.
- Any submission path that uploads before the user has seen the payload.
- Changing the coverage factor or the confidence tiers. Both were settled on 2026-08-19 and neither moves without new evidence.
