# Production hardening — design

**Date:** 2026-08-20
**Status:** approved, implementation pending
**Follows:** `2026-08-19-terminal-ux-design.md`, and the v0.1.0 release.

## Why now

v0.1.0 is published and the repository is about to be read by strangers. The
question this phase answers is not "what else could it do" but "what will break
for someone who is not the author, on a machine that is not this one".

Three findings came out of the sweep. Two of them are the same bug.

## The root cause worth naming

**A terminating signal bypasses every `Drop` guard.** This codebase leans on
`Drop` for cleanup, correctly, and it works for normal returns and for the `?`
paths that motivated it:

- `zc_bench::disk::Scratch` removes the 512 MiB benchmark file.
- `zc_tui::run::Restore` disables raw mode and leaves the alternate screen.

Neither runs when the process is terminated by a signal, because the process
does not unwind.

| Finding | Status | Consequence |
|---|---|---|
| Ctrl-C during the disk benchmark leaks 512 MiB | **Reproduced** | A hidden `.zc-bench-scratch.tmp` is left in the model directory. The disk probe is the slowest phase, so it is exactly when an impatient user interrupts, and the model volume on a low-end laptop is the full one |
| A signal during the TUI leaves the terminal raw and on the alternate screen | **Reasoned, not reproduced** | Ctrl-C is safe — crossterm reads it as a key event because raw mode disables `ISIG` — but `kill` and a closed terminal are not. The pty harness would not start the TUI under `start_new_session`, so this one is inferred from the shared root cause rather than observed |
| `zc share` truncates its URL on Windows | **Reasoned, CI-verifiable** | See below. A contributor gets an empty file, and the share loop is what closes the Phase 0 gate |

### The Windows launcher

```rust
Command::new("cmd").args(["/C", "start", ""]).arg(url)
```

Rust quotes a Windows argument only when it contains a space or tab. The URL is
fully percent-encoded and therefore contains neither, so it reaches `cmd.exe`
unquoted. `cmd.exe` treats `&` as a command separator, and the URL carries
exactly one literal `&` — the separator between the two query parameters:

```
https://github.com/OWNER/REPO/new/main?filename=...&value=%7B%22hw%22...
```

So `start` receives everything up to the `&`, the browser opens a GitHub
"new file" page with the filename filled in and **the body empty**, and
`cmd` tries to run `value=%7B...` as a second command.

## What gets built

### Track A — correctness

**A signal-safe cleanup module.** One place that knows what must happen before
the process dies, with handlers for SIGINT, SIGTERM and SIGHUP that do the
minimum and then re-raise so the exit status still reports the signal.

The binding constraint is that a signal handler may only call async-signal-safe
functions. `unlink(2)` and `write(2)` qualify; allocation, `String`, `println!`
and locking do not. Therefore:

- The scratch path is converted to a `CString` **at registration time**, while
  it is still safe to allocate, and stored in a global the handler reads.
- The terminal restore is a fixed byte literal (`\x1b[?1049l\x1b[?25h`) written
  with `write(2)`, plus a `tcsetattr` to undo raw mode.
- The handler re-raises with the default disposition so `$?` is 130 for Ctrl-C
  rather than 0, which is what every other CLI does and what scripts expect.

On Windows there are no POSIX signals; crossterm owns console-close handling
there, and the scratch file is the only concern. `ctrl_handler`-equivalent work
is **out of scope** — it would be a second unvalidated cross-platform path
written blind, which this project has a standing rule against.

**The Windows launcher.** Replace `cmd /C start` with

```rust
Command::new("rundll32").args(["url.dll,FileProtocolHandler", url])
```

A direct exec with no shell between it and the argument, so `&` is inert.

### Track B — supply chain

The dependency tree went from 12 crates to 54 when crossterm arrived, and
nothing checks it.

- `deny.toml` plus a `cargo-deny` step in CI: advisories, licence allowlist,
  and a **bans** rule asserting crossterm is reachable only from `zc-tui`. That
  last one turns the architectural promise in the terminal-UX spec into
  something CI enforces rather than something a reviewer has to remember.
- `dependabot.yml`, monthly, cargo and github-actions.

### Track C — governance

- `SECURITY.md` pointing at GitHub private vulnerability reporting. No email
  address is published; the repository setting is the user's to enable.
- `CODE_OF_CONDUCT.md` — Contributor Covenant 2.1, verbatim, with the reporting
  route pointing at the same private channel.
- `CHANGELOG.md` — Keep a Changelog format, starting at the v0.1.0 that
  shipped, written from the actual git history rather than invented.
- `.github/PULL_REQUEST_TEMPLATE.md` asking the two questions this project
  actually cares about: did `./check.sh` pass, and is every number in the
  change measured or derived from a measurement rather than assumed.

## Testing

**The leak gets a regression test that reproduces the bug**, in the manner the
bug was found: `scripts/signal_smoke.py`, a third pty-driven script beside
`tui_smoke.py` and `contract_smoke.py`. It runs `zc check` until the scratch
file exists, sends SIGINT to the child's process group, waits, and asserts the
file is gone and the exit status reports the signal. It skips cleanly where no
pty can be allocated, as its neighbours do.

It must be verified against the *unfixed* binary first — a regression test that
passes before the fix is not a regression test.

**The Windows launcher gets a unit test** asserting the constructed argument
list still carries the whole URL, `&` and everything after it included. It runs
on every platform because it inspects the arguments rather than spawning
anything, and `windows-latest` in CI therefore runs it too.

**`cargo-deny`'s bans rule is its own test**: it fails the build if a second
crate takes a crossterm dependency.

## Out of scope

Named so they are not re-litigated mid-build:

- `zc serve` (G10) and the web result cards (G11). Both are distribution
  surfaces; this phase is about correctness.
- Expanding the fuzzing. `crates/zc-model/tests/fuzz.rs` already puts 10,000
  hostile inputs through every parser that sees foreign bytes and found
  nothing. More would be speculative.
- A Windows console-close handler, for the reason given above.
- Signal handling for anything other than cleanup. `zc` has no state worth
  checkpointing and no work worth resuming.

## Risks

| Risk | Mitigation |
|---|---|
| A signal handler that is not async-signal-safe deadlocks instead of cleaning up | Only `unlink`, `write`, `tcsetattr` and `raise`. The path is allocated at registration, never in the handler |
| The Windows launcher fix is written blind | The unit test asserts the argument survives intact and runs on `windows-latest`; `rundll32` has no shell to parse `&` at all, which is why it is the fix rather than more quoting |
| `cargo-deny` becomes noise that gets ignored | Advisories fail the build; everything else is an allowlist that only changes when a dependency does |
