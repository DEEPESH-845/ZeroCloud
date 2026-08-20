# Changelog

Notable changes to `zc`. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [semantic versioning](https://semver.org/) — pre-1.0, so
the minor number carries breaking changes.

Entries name the *symptom*, because that is what a user recognises.

## [Unreleased]

### Fixed

- **Interrupting `zc check` no longer leaves 512 MiB on your disk.** The disk
  benchmark creates a scratch file when it cannot find a large existing one,
  and removed it with a `Drop` guard — which does not run when a signal kills
  the process. Ctrl-C during the benchmark, the slowest phase and so the one
  people interrupt, left a hidden `.zc-bench-scratch.tmp` in the model
  directory. Cleanup now also happens from a signal handler, and the exit
  status still reports the signal rather than pretending success.
- **`zc share` opened an empty page on Windows.** The browser was launched
  through `cmd /C start`, which treats `&` as a command separator — and the
  share URL carries exactly one, right before `value=`, which is the entire
  payload. Contributors got a GitHub "new file" page with the filename filled
  in and nothing in it. It now launches through `rundll32
  url.dll,FileProtocolHandler`, which has no shell to re-parse the argument.

### Added

- `SECURITY.md`, `CODE_OF_CONDUCT.md`, a pull-request template, and Dependabot.
- `cargo-deny` in CI: advisories, licences, sources, and a rule asserting
  `crossterm` stays reachable from `zc-tui` alone.
- `scripts/signal_smoke.py` and `crates/zc-model/tests/fuzz.rs`.

## [0.1.0] — 2026-08-20

First release. `zc` measures your machine and predicts what local models will
do on it.

### Added

- **`zc check`** — measures RAM bandwidth, compute and uncached disk, then
  predicts decode speed, time to first token and maximum context for 26 models
  across their quantisations. Opens an interactive table when a terminal is
  attached; prints plain text when piped, redirected, or given `--json`.
- **`zc plan`** — the inverse: given a model and a context length, what memory
  and how much memory *bandwidth* would running it take.
- **`zc check <hf-repo-id>`** — answers for a model the catalog does not have,
  from that repository's own published metadata. The only command that opens a
  connection, and it prints every URL before fetching it.
- **`zc verify`** — runs a model through Ollama, llama.cpp or LM Studio and
  compares the measurement against the prediction.
- **`zc share`** — turns that measurement into a pull request, without a token,
  an account, or anything leaving the machine that you have not seen first.
- **`zc fit`**, **`zc gate`**, **`zc doctor`** — the fitted coefficients and
  their evidence, the published accuracy number, and a paste-ready bug report.
- Binaries for macOS, Linux and Windows. The crates are packaged and verified
  for publishing, but nothing is on crates.io yet — the Rust route is
  `cargo install --git`.

### Known limitations

- **The Phase 0 gate is open: 1 of 2 bare-metal machines.** The accuracy number
  is computed and printed, but the dataset is thinner than it should be, and
  hypervisors run 10–30% below real hardware. `zc gate` says so on every run.
  Only a contributed physical laptop closes it.
- Time to first token is reported as `-` until a run on your backend has
  measured it. It is not derived from the CPU benchmark, because doing so was
  measured to be wrong by 10–40×.
- GPU memory bandwidth is a family table rather than a measurement, so any
  prediction resting on it is capped at `low` confidence.
- The interactive table has never been run by a human on Windows or Linux. CI
  exercises the unit tests and the non-terminal contracts on both.

[Unreleased]: https://github.com/DEEPESH-845/ZeroCloud/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DEEPESH-845/ZeroCloud/releases/tag/v0.1.0
