mod check;
mod fit_cmd;
mod machine;
mod verify;

const HELP: &str = "\
zc - what can this machine run, and how fast?

USAGE
    zc [check]            probe hardware and predict model performance
    zc verify [MODEL]     run a real model and compare against the prediction
    zc fit                show fitted coefficients and how much evidence backs them
    zc --help

Both commands run a ~20s hardware benchmark first. Nothing leaves this
machine; `zc verify` writes only to data/calibration/local.jsonl.";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("check");

    if matches!(cmd, "-h" | "--help" | "help") {
        println!("{HELP}");
        return;
    }
    // `fit` reads a file; no hardware probe needed.
    if cmd == "fit" {
        std::process::exit(fit_cmd::run());
    }
    if !matches!(cmd, "check" | "verify") {
        eprintln!("unknown command '{cmd}'\n\n{HELP}");
        std::process::exit(2);
    }

    // Fail before benchmarking, not after: a user without a runtime should not
    // wait 20 seconds to be told to install one.
    let endpoint = if cmd == "verify" {
        match verify::precheck() {
            Ok(ep) => Some(ep),
            Err(code) => std::process::exit(code),
        }
    } else {
        None
    };

    // Both subcommands need the same measured facts, and measuring twice could
    // give two different answers if thermal state shifted between them.
    let m = machine::probe();

    let fit = zc_model::Fit::load(&fit_cmd::path());

    let code = match endpoint {
        Some(ep) => verify::run(&m, &ep, &fit, args.get(1).map(String::as_str)),
        None => check::run(&m, &fit),
    };
    std::process::exit(code);
}
