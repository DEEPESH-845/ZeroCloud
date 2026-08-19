# Contributing to ZeroCloud

The most valuable contributions to this project involve no Rust at all.

## Add your machine to the calibration dataset

This is the bottleneck. Every prediction rests on `η`, a coefficient fitted
from real measured runs, and thin evidence is why every range is currently
wide. Bare metal, old hardware and unusual hardware are worth far more than
another fast laptop.

```sh
ollama pull qwen3:1.7b      # or llama.cpp / LM Studio, any model you already have
zc verify
zc share
```

`zc verify` prints predicted vs actual and appends one JSON line to
your calibration file, on your disk and nowhere else — inside a checkout that
is `data/calibration/local.jsonl`, and outside one it is your platform's data
directory (`~/Library/Application Support/zerocloud` on macOS,
`~/.local/share/zerocloud` on Linux, `%LOCALAPPDATA%\zerocloud` on Windows).
Every command that touches it prints the full path. `zc share` then
shows you that line field by field, tells you what is not in it, and builds a
GitHub URL with the record prefilled — asking before it hands that URL to your
browser. `zc` opens no connection itself; your browser does, and you watch it.
`zc share --print` gives you the URL without the prompt.

Committing through GitHub's editor forks the repository to your account and
offers the pull request button. The file lands in `data/calibration/community/`
under a name derived from its own contents, so resubmitting the same run is an
empty diff rather than a second PR.

Read the record before you send it. It contains your hardware fingerprint (a
hash), OS, virtualization kind, measured bandwidths, the model and
quantisation, and the prediction that was made *before* the run. It contains no
hostname, username, serial, MAC, IP or file path. If you would rather not
publish your bandwidth figures, that is a completely reasonable choice — say no
at the prompt.

You can run the same check CI runs, before opening anything:

```sh
python3 scripts/validate_calibration.py
```

It recomputes `error_pct` and `within_range` from the record's own numbers, so
a record that disagrees with itself fails here rather than in review.

Merged records feed `zc fit` straight away, which is what makes your machine
improve everyone's predictions. They do not move the headline accuracy figure —
that is computed from `data/calibration/gate.jsonl`, the tier whose provenance
is known, and promotion into it is a maintainer's deliberate `git mv`. `zc
gate` prints both numbers.

**A record whose prediction was badly wrong is the most useful record there
is.** It will not be rejected for making the number worse. The only records
ever retired are ones whose provenance is invalid, and each retirement is
written down in `data/calibration/archive/README.md` with its reason.

## Add or correct a model

One JSON file in `data/models/`, one PR, no Rust. Copy the nearest existing
file and change the numbers; `crates/zc-model/build.rs` picks it up
automatically.

Every field must come from the model's own `config.json` — layer count,
`n_embd`, vocabulary size, and the attention geometry (`n_kv_heads`,
`head_dim`). KV cache size varies by 4–8x between MHA and GQA, so a guessed
`n_kv_heads` produces a confidently wrong context number. If a field is not in
`config.json`, leave the model out rather than estimating it.

You can also drop a JSON file into your own config directory
(`~/.config/zerocloud/models/` on Linux, `~/Library/Application
Support/zerocloud/models/` on macOS, `%APPDATA%\zerocloud\models\` on Windows)
to test it without rebuilding.

## Report a wrong prediction

Open an issue with the output of `zc verify` and `zc doctor`. `zc doctor` is
designed for exactly this: it prints every raw probe reading next to the value
derived from it, so a wrong prediction can be traced to the measurement that
caused it without a round trip. It carries no identifying information.

## Code

```sh
./check.sh    # tests, clippy, cross-target clippy, installer self-check
```

`check.sh` must be green before every commit. It cross-compiles against Linux
and Windows, because a `#[cfg(target_os)]` block that only breaks on Windows is
otherwise invisible on a Mac — that has cost this project days.

Three rules that are not negotiable, because they are the product:

1. **A number is measured, derived from measured inputs, or printed as `-`.**
   If your change makes something fall back to a plausible constant when the
   input is unmeasurable, that is the bug. A missing number is recoverable; a
   confidently wrong one is not.
2. **Coefficients move by `zc fit` from calibration records, never by hand** —
   including when the gate is red. Widening a range honestly is always
   allowed; narrowing one to make a number pass is never allowed.
3. **`zc-probe`, `zc-bench` and `zc-model` stay dependency-free**, including
   dev-dependencies. Tests use `std` only. `zc-cli` has exactly one
   target-gated dependency (`libc`, for the default SIGPIPE disposition) and
   `zc-runtime` speaks HTTP by hand. A single binary with no supply chain is
   worth more than the convenience of any crate that would end that.

Tests assert a hand-computed physical quantity or lock in a specific bug, with
the reasoning in the doc comment. A test that only restates the implementation
is noise.

## Scope

`zc` predicts and measures. It does not run models — Ollama, llama.cpp and LM
Studio do, and `zc` reads their reported timings. Changes that would make `zc`
an inference engine are out of scope for now; see `PLAN.md` Phase 5.
