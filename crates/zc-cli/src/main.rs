mod check;
mod fit_cmd;
mod gate_cmd;
mod machine;
mod verify;

const HELP: &str = "\
zc - what can this machine run, and how fast?

USAGE
    zc [check] [--json]   probe hardware and predict model performance
    zc verify [MODEL] [--runtime NAME]
                          run a real model and compare against the prediction
    zc fit                show fitted coefficients and how much evidence backs them
    zc gate               how wrong have we been? (exits non-zero until it passes)
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
    AGENT USAGE    `zc check --json` emits one object:
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

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // Global flags are stripped before dispatch so they may appear anywhere.
    let as_json = take_flag(&mut args, "--json");
    let runtime = take_value(&mut args, "--runtime");
    let cmd = args.first().map(String::as_str).unwrap_or("check");

    if matches!(cmd, "-h" | "--help" | "help") {
        println!("{HELP}");
        return;
    }
    // `fit` and `gate` read a file; no hardware probe needed.
    if cmd == "fit" {
        std::process::exit(fit_cmd::run());
    }
    if cmd == "gate" {
        std::process::exit(gate_cmd::run());
    }
    if !matches!(cmd, "check" | "verify") {
        eprintln!("unknown command '{cmd}'\n\n{HELP}");
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
        Some(rt) => verify::run(&m, rt.as_ref(), &fit, args.get(1).map(String::as_str)),
        None => check::run(&m, &fit, as_json),
    };
    std::process::exit(code);
}

/// Remove `flag` and the value after it, if present.
fn take_value(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let i = args.iter().position(|a| a == flag)?;
    args.remove(i);
    (i < args.len()).then(|| args.remove(i))
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
