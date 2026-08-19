mod check;
mod doctor;
mod fit_cmd;
mod gate_cmd;
mod hf;
mod machine;
mod share;
mod verify;

const HELP: &str = "\
zc - what can this machine run, and how fast?

EXAMPLES
    zc                    browse what this machine can run  (arrows, ? for keys)
    zc check --top 5      the five best fits, as plain text
    zc check --json       the same data, for a script or an agent
    zc verify qwen3:1.7b  run the model for real and compare
    zc check Qwen/Qwen3-4B  will a model outside the catalog fit? (fetches)
    zc doctor             a paste-ready report for a bug

USAGE
    zc [check] [--json] [--kv f16|q8|q4] [--top N] [--all] [--tui | --no-tui]
                          probe hardware and predict model performance
    zc check <hf-repo-id> will one model that is not in the catalog fit?
                          the only command that touches the network
    zc verify [MODEL] [--runtime NAME]
                          run a real model and compare against the prediction
    zc fit                fitted coefficients, and the evidence behind them
    zc gate               how wrong have we been? (non-zero until it passes)
    zc share [--record FILE] [--print]
                          submit your last `zc verify` measurement upstream
    zc doctor             everything probed, measured and concluded, as Markdown
    zc --help

Both commands benchmark the hardware first: ~2s on a fast laptop, longer
where the disk is slow. Nothing leaves this machine and no connection is
opened, with one exception you have to ask for by name: `zc check
<hf-repo-id>` reads that repo's public metadata from huggingface.co, and
prints each URL before fetching it. `zc verify` writes only to
crates/zc-model/data/calibration/local.jsonl.

zc check
    PRECONDITIONS  none. No runtime and no model files required, and no
                   network unless you pass an <hf-repo-id>.
    SIDE EFFECTS   reads a large existing file on the model volume to measure
                   uncached disk. Creates a scratch file only if none is found.
                   Runs nvidia-smi / lspci / powershell read-only to find GPUs,
                   each abandoned after 4s.
    EXIT CODES     0 always.
    AGENT USAGE    One row per model by default: the highest-fidelity
                   quantisation with the best verdict, since bytes per
                   parameter is measured and model quality is not.
                   Rows are ranked verdict, then decode speed, then context -
                   every term measured or derived, never a blended score.
                   `--all` lists every quantisation; `--top N` (default 20)
                   sets the row limit and applies on top of it, so
                   `--all --top 3` is three rows drawn from every
                   quantisation. The cut applies to `--json` identically, and
                   `.assumptions.total_rows` reports how many rows existed
                   before it.
    INTERACTIVE    On a terminal, `zc check` opens a browsable table:
                   arrows or j/k move, enter shows how a number was derived,
                   / filters by name, s sorts, a toggles every quantisation,
                   ? lists the keys, q quits and leaves the report in your
                   scrollback. Piped, redirected, or with --json it prints
                   plain text instead -- byte for byte what it always has.
                   --tui forces it on where the terminal is not detected,
                   --no-tui forces it off, and ZC_ASCII=1 swaps the
                   box-drawing glyphs for ASCII on terminals that need it.
                   `zc check --json` emits one object:
                   .schema                     integer, bumped when this breaks
                   .machine.{cpu,memory,env,storage,gpus,budget}
                   .machine.storage.{model_dir,mount}
                                               home-relative (`~/...`), so a
                                                report you attach carries no
                                                account name. Schema 2.
                   .machine.gpus[]             name, vendor, vram_bytes (0 for
                                                integrated - it shares system
                                                memory rather than adding any),
                                                bw_gbs, bw_measured
                   .measured.{ram,compute,disk} disk is null if unmeasurable
                   .assumptions                backend, bandwidths, prompt len
                   .models[]                    id, quant, verdict, decode_tok_s
                                                {low,high}, max_context, ttft_s,
                                                confidence, resident_fraction
                   verdict is one of good|usable|slow|wont_fit.
                   ttft_s and prefill_tok_s are null until `zc verify` has
                   measured this backend - they are never derived.

zc doctor
    PRECONDITIONS  none. Same probe and benchmark as `zc check`.
    SIDE EFFECTS   same as `zc check`.
    EXIT CODES     0 always.
    AGENT USAGE    Markdown for a bug report: raw probe readings alongside the
                   struct derived from them, the full bandwidth-vs-threads
                   curve, and the fitted coefficients. Carries no hostname,
                   username, serial, MAC or IP; paths are rewritten to `~`.

zc verify [MODEL] [--runtime NAME]
    PRECONDITIONS  a local runtime that reports its own prefill/decode timings:
                   ollama (:11434), llamacpp (:8080), lmstudio (:1234).
                   Override with OLLAMA_HOST, LLAMA_SERVER_HOST or
                   LMSTUDIO_HOST.
                   vLLM, MLX and Docker Model Runner are detected and reported
                   but refused: their APIs report no timing split, so a rate
                   measured through them would include HTTP and scheduling.
    SIDE EFFECTS   runs one warm-up and one measured generation, then appends
                   one line to data/calibration/local.jsonl. Nothing is sent
                   anywhere.
    EXIT CODES     0 measured, 1 runtime error, 2 nothing usable to measure.
    AGENT USAGE    MODEL is a name prefix; the smallest installed model is the
                   default. Ollama can measure any model it has; the others
                   report no attention geometry, so their models must match a
                   catalog entry or the run is refused rather than guessed.";

/// Version, printed by `--version`. Comes from the workspace `Cargo.toml`, so
/// a released binary can never disagree with the tag it was built from.
const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    // Rust ignores SIGPIPE so writes fail with EPIPE, and `println!` panics on
    // a failed write. `zc check | head` therefore ended in a panic message and
    // a backtrace hint -- for a tool whose output is meant to be piped into
    // `head`, `less` and `grep`, that reads as a crash. Restoring the default
    // disposition makes the process exit quietly when the reader goes away,
    // which is what every other CLI does.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // Global flags are stripped before dispatch so they may appear anywhere.
    let as_json = take_flag(&mut args, "--json");
    let runtime = take_value(&mut args, "--runtime");
    let kv_arg = take_value(&mut args, "--kv");
    let kv = match &kv_arg {
        Some(v) => match zc_model::KvPrecision::parse(v) {
            Some(p) => p,
            None => {
                eprintln!("unknown --kv value '{v}' (expected f16, q8 or q4)");
                std::process::exit(2);
            }
        },
        None => zc_model::KvPrecision::DEFAULT,
    };
    // Default cut. The catalog is large enough now that printing all of it
    // buries the models a constrained machine can actually run under the ones
    // it cannot.
    const DEFAULT_TOP: usize = 20;
    // `--all` means both: every quantisation, and no row limit.
    let show_all = take_flag(&mut args, "--all");
    let top_arg = take_value(&mut args, "--top");
    let top = match &top_arg {
        Some(v) => match v.parse::<usize>() {
            Ok(n) if n > 0 => Some(n),
            _ => {
                eprintln!("--top needs a positive number, got '{v}'");
                std::process::exit(2);
            }
        },
        None if show_all => None,
        None => Some(DEFAULT_TOP),
    };
    // `share` reads a record file rather than measuring, so both of its flags
    // are stripped here with the other globals and never reach the probe.
    let record = take_value(&mut args, "--record");
    let print_only = take_flag(&mut args, "--print");
    let force_tui = take_flag(&mut args, "--tui");
    let no_tui = take_flag(&mut args, "--no-tui");
    let cmd = args.first().map(String::as_str).unwrap_or("check");

    if matches!(cmd, "-h" | "--help" | "help") {
        println!("{HELP}");
        return;
    }
    if matches!(cmd, "-V" | "--version" | "version") {
        println!("zc {VERSION}");
        return;
    }
    // Anything still starting with `-` was not a flag we know. Silently
    // ignoring it is the worse failure: `zc check --josn` would print a normal
    // report and the user would never learn the flag did nothing.
    if let Some(bad) = args.iter().find(|a| a.starts_with('-')) {
        eprintln!("unknown option '{bad}' -- run `zc --help`");
        std::process::exit(2);
    }
    // A known flag on a command that ignores it fails exactly the way an
    // unknown flag did: `zc doctor --json` printed Markdown and exited 0, so
    // an agent piping it into `jq` got a parse error rather than being told
    // the flag does not apply there. Same rule, same message shape.
    let supplied = [
        ("--json", as_json),
        ("--runtime", runtime.is_some()),
        ("--kv", kv_arg.is_some()),
        ("--top", top_arg.is_some()),
        ("--all", show_all),
        ("--record", record.is_some()),
        ("--print", print_only),
        ("--tui", force_tui),
        ("--no-tui", no_tui),
    ];
    for (flag, present) in supplied {
        if present && !accepts(cmd, flag) {
            eprintln!("`zc {cmd}` does not take {flag} -- run `zc --help`");
            std::process::exit(2);
        }
    }

    // `fit` and `gate` read a file; no hardware probe needed.
    if cmd == "fit" {
        std::process::exit(fit_cmd::run());
    }
    if cmd == "gate" {
        std::process::exit(gate_cmd::run());
    }
    if cmd == "share" {
        std::process::exit(share::run(record.as_deref(), print_only));
    }
    if !matches!(cmd, "check" | "verify" | "doctor") {
        match did_you_mean(cmd) {
            Some(c) => eprintln!("unknown command '{cmd}' -- did you mean `zc {c}`?"),
            None => eprintln!("unknown command '{cmd}' -- run `zc --help`"),
        }
        std::process::exit(2);
    }

    // Fail before benchmarking, not after: a user without a runtime should not
    // sit through the benchmark only to be told to install one.
    let runtime = if cmd == "verify" {
        match verify::precheck(runtime.as_deref()) {
            Ok(rt) => Some(rt),
            Err(code) => std::process::exit(code),
        }
    } else {
        None
    };

    // The TUI opens only when a human is at both ends of the pipe. Every other
    // path -- a pipe, a redirect, --json, CI, an agent -- takes the static
    // renderer and gets exactly what it always got. stdin matters as much as
    // stdout: raw mode needs a real terminal to read keys from, and `zc < /dev/null`
    // would otherwise open a screen nobody can drive.
    let interactive = std::io::IsTerminal::is_terminal(&std::io::stdout())
        && std::io::IsTerminal::is_terminal(&std::io::stdin())
        && !as_json
        && !no_tui
        && !std::env::var("TERM").is_ok_and(|t| t == "dumb");
    // Requested and impossible is an error, never a silent downgrade -- the
    // standing rule about substituting a fallback for the thing that was asked
    // for applies to the interface as much as to a measurement.
    if force_tui && !interactive {
        eprintln!("--tui needs an interactive terminal on both stdin and stdout");
        std::process::exit(2);
    }
    // A repository id after `check` asks about a model the catalog does not
    // have. It is the one path that opens an outbound connection, so it never
    // opens the TUI -- the fetch and its two URLs must be visible.
    let hf_repo = args
        .get(1)
        .filter(|_| cmd == "check")
        .filter(|a| hf::looks_like_repo_id(a))
        .cloned();
    let tui = cmd == "check" && interactive && hf_repo.is_none();

    // Both subcommands need the same measured facts, and measuring twice could
    // give two different answers if thermal state shifted between them.
    let m = machine::probe();

    let fit = zc_model::Fit::from_text(&fit_cmd::read_text(&fit_cmd::sources()));

    let code = match runtime {
        Some(rt) => verify::run(&m, rt.as_ref(), &fit, kv, args.get(1).map(String::as_str)),
        None if cmd == "doctor" => doctor::run(&m, &fit, kv),
        None if hf_repo.is_some() => {
            if as_json {
                eprintln!("`zc check <hf-repo-id>` has no --json output yet -- it reports memory");
                eprintln!("only, and the shape is not settled. Drop --json, or use a catalog id.");
                std::process::exit(2);
            }
            check::run_hf(&m, kv, &hf_repo.unwrap())
        }
        None => check::run(&m, &fit, kv, top, show_all, as_json, tui),
    };
    std::process::exit(code);
}

/// The closest command name, if one is close enough to be worth suggesting.
///
/// Plain Levenshtein over a six-item list; a fuzzy-match dependency for this
/// would be absurd. The distance cap is the point: suggesting `share` for
/// `xyzzy` is worse than saying nothing at all.
fn did_you_mean(input: &str) -> Option<&'static str> {
    const CMDS: &[&str] = &["check", "verify", "fit", "gate", "share", "doctor"];
    CMDS.iter()
        .map(|c| (*c, distance(input, c)))
        .filter(|(c, d)| *d <= 2 && *d < c.len())
        .min_by_key(|(_, d)| *d)
        .map(|(c, _)| c)
}

fn distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let sub = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + sub);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Which flags each command actually reads.
///
/// Kept as one table rather than a check at each use site, because the failure
/// this prevents is a *missing* check: every flag is stripped globally so it
/// may appear anywhere, and a command that never looks at one would otherwise
/// accept it in silence.
fn accepts(cmd: &str, flag: &str) -> bool {
    let ok: &[&str] = match cmd {
        "check" => &["--json", "--kv", "--top", "--all", "--tui", "--no-tui"],
        "verify" => &["--runtime", "--kv"],
        "doctor" => &["--kv"],
        "share" => &["--record", "--print"],
        // `fit` and `gate` read the calibration file and report it. Neither
        // predicts anything, so no prediction flag applies.
        _ => &[],
    };
    ok.contains(&flag)
}

/// Remove `flag` and the value after it, if present.
///
/// A flag with nothing after it is an error, not an absent flag: `zc check
/// --top` used to fall through to the default of 20, so a user who meant
/// `--top 50` got 20 rows and no indication that anything was wrong.
fn take_value(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let i = args.iter().position(|a| a == flag)?;
    args.remove(i);
    if i >= args.len() {
        eprintln!("{flag} needs a value");
        std::process::exit(2);
    }
    Some(args.remove(i))
}

/// Remove `flag` from `args` if present, reporting whether it was there.
fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    match args.iter().position(|a| a == flag) {
        Some(i) => {
            args.remove(i);
            true
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    /// Every flag the help text lists for a command must be accepted by it,
    /// and flags it does not read must be refused. `zc doctor --json` used to
    /// print Markdown and exit 0.
    /// clig.dev: if the user did something wrong and you can guess what they
    /// meant, suggest it.
    #[test]
    fn a_near_miss_command_is_suggested() {
        assert_eq!(super::did_you_mean("chekc"), Some("check"));
        assert_eq!(super::did_you_mean("verift"), Some("verify"));
        assert_eq!(super::did_you_mean("doctr"), Some("doctor"));
        assert_eq!(super::did_you_mean("gat"), Some("gate"));
        assert_eq!(super::did_you_mean("shre"), Some("share"));
        // Nothing close enough is worse than no suggestion at all.
        assert_eq!(super::did_you_mean("xyzzy"), None);
        assert_eq!(super::did_you_mean(""), None);
    }

    #[test]
    fn flags_are_scoped_to_the_commands_that_read_them() {
        for (cmd, flag) in [
            ("check", "--json"),
            ("check", "--all"),
            ("check", "--tui"),
            ("check", "--no-tui"),
            ("verify", "--runtime"),
            ("verify", "--kv"),
            ("doctor", "--kv"),
            ("share", "--record"),
            ("share", "--print"),
        ] {
            assert!(super::accepts(cmd, flag), "{cmd} should accept {flag}");
        }
        for (cmd, flag) in [
            ("doctor", "--json"),
            ("doctor", "--top"),
            ("doctor", "--tui"),
            ("gate", "--no-tui"),
            ("fit", "--json"),
            ("fit", "--kv"),
            ("gate", "--json"),
            ("share", "--json"),
            ("check", "--runtime"),
            ("check", "--print"),
            ("verify", "--top"),
        ] {
            assert!(!super::accepts(cmd, flag), "{cmd} must refuse {flag}");
        }
    }
}
