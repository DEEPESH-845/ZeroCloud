mod check;
mod doctor;
mod fit_cmd;
mod gate_cmd;
mod machine;
mod verify;

const HELP: &str = "\
zc - what can this machine run, and how fast?

USAGE
    zc [check] [--json] [--kv f16|q8|q4] [--top N | --all]
                          probe hardware and predict model performance
    zc verify [MODEL] [--runtime NAME]
                          run a real model and compare against the prediction
    zc fit                show fitted coefficients and how much evidence backs them
    zc gate               how wrong have we been? (exits non-zero until it passes)
    zc doctor             everything probed, measured and concluded, as Markdown
    zc --help

Both commands run a ~20s hardware benchmark first. Nothing leaves this
machine; `zc verify` writes only to data/calibration/local.jsonl.

zc check
    PRECONDITIONS  none. No network, no runtime, no model files required.
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
                   `--all` lists every quantisation and lifts the row limit;
                   `--top N` (default 20) sets it. The cut applies to `--json`
                   identically, and `.assumptions.total_rows` reports how many
                   rows existed before it.
                   `zc check --json` emits one object:
                   .schema                     integer, bumped on breaking change
                   .machine.{cpu,memory,env,storage,gpus,budget}
                   .machine.gpus[]             name, vendor, vram_bytes (0 for
                                                integrated - it shares system
                                                memory rather than adding any),
                                                bw_gbs, bw_measured
                   .measured.{ram,compute,disk} disk is null if unmeasurable
                   .assumptions                 backend, bandwidths, prompt length
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
                   Override with OLLAMA_HOST / LLAMA_SERVER_HOST / LMSTUDIO_HOST.
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
    let kv = match take_value(&mut args, "--kv") {
        Some(v) => match zc_model::KvPrecision::parse(&v) {
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
    let top = match take_value(&mut args, "--top") {
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
    // `fit` and `gate` read a file; no hardware probe needed.
    if cmd == "fit" {
        std::process::exit(fit_cmd::run());
    }
    if cmd == "gate" {
        std::process::exit(gate_cmd::run());
    }
    if !matches!(cmd, "check" | "verify" | "doctor") {
        eprintln!("unknown command '{cmd}' -- run `zc --help`");
        std::process::exit(2);
    }

    // Fail before benchmarking, not after: a user without a runtime should not
    // wait 20 seconds to be told to install one.
    let runtime = if cmd == "verify" {
        match verify::precheck(runtime.as_deref()) {
            Ok(rt) => Some(rt),
            Err(code) => std::process::exit(code),
        }
    } else {
        None
    };

    // Both subcommands need the same measured facts, and measuring twice could
    // give two different answers if thermal state shifted between them.
    let m = machine::probe();

    let fit = zc_model::Fit::load(&fit_cmd::path());

    let code = match runtime {
        Some(rt) => verify::run(&m, rt.as_ref(), &fit, kv, args.get(1).map(String::as_str)),
        None if cmd == "doctor" => doctor::run(&m, &fit, kv),
        None => check::run(&m, &fit, kv, top, show_all, as_json),
    };
    std::process::exit(code);
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
