# Terminal UX — design

**Date:** 2026-08-19
**Status:** approved, implementation pending
**Supersedes nothing.** Extends the launch surface built in
`2026-08-17-gate-closure-and-launch-design.md`.

## Why

`zc` prints a correct report and prints it badly. Three defects, all found by
running the release binary from outside the repo:

1. **The terminal is dead for the whole benchmark.** First byte and last byte
   both land at 1.88 s on an M5 with a 5 GB/s SSD. The target market is
   DRAM-less drives and 5400 rpm disks, where the disk probe alone runs far
   longer. [clig.dev](https://clig.dev/) states the rule plainly: print
   something inside 100 ms or the program looks broken. `zc` looks broken
   longest on exactly the hardware it exists to serve.
2. **Rows overflow to 93 columns** whenever a model is not fully resident —
   again, precisely the low-end machine this product targets. The cause is a
   fixed 28-character model column that the widest catalog id happens to need.
3. **There is no interactive surface at all.** `llmfit`, the direct competitor,
   defaults to a TUI with a ranked table and a community leaderboard. `zc` is
   batch-only.

The first two are defects. The third is a competitive gap, and the interesting
part is that closing it plays to a strength `llmfit` cannot copy.

## The differentiator

`llmfit` shows a score. `zc` measured the machine, so `zc` can show the
*derivation*. That is the entire product thesis — every number is measured,
derived from measured inputs, or printed as a dash — and a detail pane is the
first surface where a user can interrogate it directly:

```
  llama-3.1-8b · Q8_0                            10.3-17.2 tok/s

  weights           8.54 GiB          84% resident in RAM
  spill             1.37 GiB  @  5.0 GB/s disk
  bandwidth          125 GB/s  measured, 10 threads
  η                    0.875  from 4 runs on 1 machine
  confidence             low  ← 1 machine; 8 needed for medium
  max context          37K    KV f16, 0.50 MiB/token
```

Nobody else can print that, because nobody else measured the machine. This is
the reason to build a TUI, and it is what keeps the build from being decoration.

## Decisions taken

| Question | Decision |
|---|---|
| Interactive or static? | **TUI by default** on a TTY, static everywhere else |
| Which commands? | **`check` only** (and bare `zc`). `verify`/`fit`/`gate`/`doctor`/`share` stay static |
| TUI layer | **crossterm only**, no ratatui |
| Glyphs | **Unicode with automatic ASCII fallback** |
| Benchmark feedback | **Live progress resolving in place**, on stderr |

### Why crossterm and not ratatui

Measured, not estimated:

| Approach | Transitive crates (Unix / Windows) |
|---|---|
| ratatui | 70 / 64 |
| crossterm alone | 28 / 23 |
| current tree | 12 external |

`zc-report/src/text.rs` is already a renderer with tested contracts — colour
never changes a column's width, no trailing whitespace on any line, dashes for
unmeasured values. ratatui means rewriting that into widgets and discarding
those tests. crossterm buys only the part that is genuinely hard *and*
untestable here: Windows console modes, key-event parsing, resize. Per
`VERIFICATION.md` the Windows paths are never executed locally, so writing that
layer blind is the one trade worth paying a dependency to avoid.

## Architecture

### Crate layout

New crate `crates/zc-tui`, depending on `crossterm`, `zc-report`, `zc-model`,
`zc-probe`, `zc-bench`. It is the **only** crate that may name crossterm.
`zc-report`, `zc-model`, `zc-probe` and `zc-bench` stay dependency-free, which
keeps the crates.io path from getting worse than it already is (see
`docs/publishing.md`).

`zc-cli` gains a dependency on `zc-tui` and nothing else changes about it.

### Mode selection — the contract that must not break

```
TUI  ⟺  stdout is a TTY
      ∧  stdin  is a TTY
      ∧  no --json
      ∧  cmd ∈ {bare, check}
      ∧  TERM ≠ dumb
      ∧  --no-tui absent
```

Everything else — a pipe, a redirect, `--json`, CI, an agent, `TERM=dumb` —
takes today's static path and emits **byte-identical output**. This is a hard
requirement, not a goal: `zc check | head`, `zc check --json | jq`,
`calibrate.yml` and every documented agent usage in `HELP` depend on it.

Two new flags on `check`:

- `--tui` forces the TUI on. For containers and odd terminals, matching what
  `llmfit` offers for the same reason.
- `--no-tui` forces it off.

`--tui` on a non-TTY is **an error, exit 2** — never a silent downgrade.
Silently substituting a fallback when the requested thing is impossible is the
failure mode this codebase already has a standing rule against.

Both flags go in `accepts()` in `main.rs` under `check` only, so
`zc doctor --tui` is refused the same way `zc doctor --json` now is.

### Progress

A `progress` module in `zc-tui`, used by **both** paths.

- Writes to **stderr**, and only when stderr is a TTY. So `zc check > out.txt`
  shows the user progress while `out.txt` stays clean, and a non-TTY stderr
  (CI logs) stays silent.
- One line per measurement — ram, compute, disk. Printed on start, animated
  while running, rewritten in place with the measured result on completion.
- Redraw uses `\r` and a clear-to-end-of-line, never cursor-up, so a resize
  mid-benchmark cannot corrupt earlier lines.

This changes `zc check 2>&1 | head` output. That is intended and is an
improvement; nothing machine-readable reads stderr.

### Charset

A `charset` module in **`zc-report`** — not `zc-tui` — because the static
renderer needs the same glyphs and `zc-report` must stay dependency-free.
Resolves **once** at startup to `Unicode` or `Ascii`:

- `ZC_ASCII=1` → Ascii, unconditionally. Escape hatch, and what tests set.
- `TERM=dumb` → Ascii.
- Unix: Unicode only if `LC_ALL`, `LC_CTYPE` or `LANG` (first one set wins,
  in that order) contains `UTF-8` or `utf8`, case-insensitively.
- Windows: Unicode only if `WT_SESSION` (Windows Terminal) or `TERM` (Git Bash,
  WSL, MSYS) is set. Otherwise Ascii — which is the correct answer for the
  legacy `conhost` console still shipping on the old laptops this targets.

Detection is **environment variables only, no system calls**. Probing
`GetConsoleOutputCP` would mean an `extern "system"` block in a crate that has
none, running on a platform `VERIFICATION.md` records as never executed here —
reintroducing precisely the untestable unsafe code that the crossterm
dependency was chosen to avoid. Guessing Ascii on an unknown Windows console
degrades to something that renders everywhere; guessing Unicode wrong prints
replacement boxes.

Every glyph is looked up through it. No literal box-drawing character appears
anywhere else in the codebase.

| Role | Unicode | Ascii |
|---|---|---|
| fully resident | `●` | `*` |
| partly resident | `◐` | `o` |
| won't fit | `○` | `.` |
| horizontal rule | `─` | `-` |
| vertical rule | `│` | `\|` |
| spinner | `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` | `-\|/` |
| separator | `·` | `-` |

Emoji are excluded deliberately: they are double-width, which misaligns every
table, and they render as replacement boxes on the old Windows consoles that
are a primary target.

### The TUI

Single screen, alternate-screen buffer, three regions.

```
┌ header ─────────────────────────────────────────────┐
│ zc 0.1.0   Apple M5 · 16 GiB unified · Metal        │
│ 125 GB/s ram · 5.0 GB/s disk · 367 GFLOPS           │
├ body ───────────────────────────────────────────────┤
│   MODEL           QUANT   DECODE       CTX    TTFT  │
│ ● qwen3-1.7b      Q8_0    48-80 t/s    40K    0.9s  │
│ ● qwen2.5-3b      Q8_0    24-41 t/s    32K    1.3s  │
│ ◐ llama-3.1-8b    Q8_0    10-17 t/s    37K    3.4s  │
├ footer ─────────────────────────────────────────────┤
│ ↑↓ move · enter why · / filter · s sort · ? help    │
└─────────────────────────────────────────────────────┘
```

Keys:

| Key | Action |
|---|---|
| `↑` `↓` `j` `k` | move cursor |
| `PgUp` `PgDn` `Home` `End` | move by page / to ends |
| `enter` | toggle the detail pane for the selected row |
| `/` | filter by model name; `esc` clears |
| `s` | cycle sort: verdict → decode → context → verdict |
| `a` | toggle all-quantisations (the `--all` view) |
| `?` | help overlay listing every key |
| `q` `esc` `ctrl-c` | quit |

The detail pane renders the derivation shown above. Every field in it is
already computed by `zc-model`; none of it is new maths, and any value that is
unmeasured prints as `-` exactly as it does in the table.

**On quit**, leave the alternate screen and then print the static report to
stdout. Scrollback keeps the answer rather than a blank screen. This is what
makes a default-on TUI acceptable in a tool people run inside other workflows.

### Shared column widths

The model column becomes the width of the widest **visible** row, clamped to
`[12, 28]`, computed once per render and shared by the static and TUI
renderers. This is the fix for the 93-column overflow: a constrained machine
shows short model ids, so the table narrows to fit rather than wrapping.

The `resident` suffix keeps its wording. Nothing is truncated — a name that
needs more than 28 characters overflows its column rather than losing
characters, because a truncated model id is not a model id.

## Isolation and testing

State and behaviour are pure and TTY-free. Only the event loop and the writer
touch crossterm.

```rust
struct State { rows, cursor, filter, sort, show_all, pane, viewport }
fn on_key(&mut State, Key) -> Action        // pure
fn frame(&State, w: u16, h: u16) -> Vec<String>  // pure
```

Unit-tested without a terminal:

- filter narrows the row set and resets the cursor into range
- sort cycles and is stable
- cursor cannot leave the row set; page moves clamp at both ends
- viewport scrolls to keep the cursor visible, including after a resize that
  shrinks the body below the cursor's position
- `frame` never emits a line wider than the width it was given
- `frame` emits no trailing whitespace on any line — the existing rule
- charset fallback maps every glyph, with `ZC_ASCII=1` forcing it
- ASCII and Unicode frames have identical line counts and identical widths

Integration-level, without a TTY:

- non-TTY stdout produces byte-identical output to the pre-change binary
- `--tui` on a non-TTY exits 2
- `--json` never opens a TUI even on a TTY

## Out of scope

Named so they are not re-litigated mid-build:

- Config file or themes. No value that never changes gets a config knob.
- Mouse support.
- A TUI for `verify`, `fit`, `gate`, `doctor` or `share`. They are one-shot
  reports and CI surfaces; `gate` and `fit` already fit one screen.
- A community leaderboard view. `zc share` covers contribution, and a
  leaderboard needs a server this project deliberately does not have.

## Risks

| Risk | Mitigation |
|---|---|
| TUI-by-default breaks a script | Mode selection requires a TTY on both stdin and stdout; every non-TTY path is byte-identical and tested as such |
| Windows terminal misbehaves | crossterm owns that layer; it is the most-exercised Windows terminal crate in Rust, which is the reason for the dependency |
| Dependency tree grows 12 → 28 | Confined to `zc-tui`; every other crate stays dependency-free |
| Binary size grows | Measure before and after and record it. Not estimated — the project rule applies to its own artifacts |
| Alternate screen loses the report | Static report printed to stdout on quit |
