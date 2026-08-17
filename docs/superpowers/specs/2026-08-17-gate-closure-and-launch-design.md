# Gate closure and launch — design

Date: 2026-08-17
Status: approved

## Problem

`zc gate` is red, and no feature closes it:

```
1 run on 1 machine      median |error| 28.3%     within range 0.0%
BLOCKED  1 of 5 machines · 0 of 2 bare-metal · 28.3% exceeds the 25% ceiling
```

One calibration record backs the product's entire claim. It predates the `virt`
field, the 1.645σ coverage factor, the 0.40 prior floor, and the f16 KV default,
so it grades a prediction from a superseded model and its provenance is unknown.

PLAN.md's Phase 0 gate — median decode error < 25% across ≥5 machines — is the
real milestone and it is a *data* problem. The llmfit parity plan
(`~/.claude/plans/i-am-providing-you-robust-donut.md`) has steps 1–5 of 9 done;
everything remaining is either launch surface or post-launch width.

## Decisions taken

- Gate closes on machines the maintainer can reach directly (5+ available), so
  the public release ships from a **green** gate with a real number in the README.
- `zc share` is therefore *not* on the critical path. It is the Phase 1 scaling
  mechanism for the <15% target, not a launch requirement.
- Launch cut is **README + release binaries + install.sh**. Nothing else.
- Distribution comes *before* gate closure: GitHub runners are natively Windows,
  Linux and macOS, so the release pipeline eliminates local cross-compilation
  rather than duplicating it. Every gate record is then produced by the exact
  artifact users download.

## Standing constraint

Unchanged and load-bearing: **a number is measured, derived from measured inputs,
or printed as `-`.** Coefficients move by `zc fit` from records, never by hand,
including when the gate is red and a hypothesis about their direction exists.

---

## Phase A — Close the Phase 0 gate

### A1. Retire the stale calibration record

`gate.rs:81` computes `slot.1 |= r.virt.as_deref() != Some("none")`, so a single
record with an absent `virt` marks that machine as virtualized permanently. The
behaviour is correct and test-locked (`unknown_virt_is_not_assumed_bare_metal`,
`one virtualized run taints the machine`), but it means the existing record bars
this Mac from ever counting toward `MIN_BARE_METAL`.

The record is retired for two stated reasons, neither of which is "it was
inconvenient": its `virt` is genuinely unknown, and it grades a superseded
prediction model.

- Move `data/calibration/local.jsonl`'s single line to
  `data/calibration/archive/2026-08-16-pre-coverage-factor.jsonl`.
- Add `data/calibration/archive/README.md` stating why each archived record left
  the active set. Archived records are history; they are never read by `zc gate`.
- Run `zc verify` on this machine to produce the first record carrying
  `virt:"none"`.

`zc fit` and `zc gate` both resolve a **single file** through `fit_cmd::path()`
(`ZC_CALIBRATION` or `data/calibration/local.jsonl`) — no directory scan — so an
`archive/` subdirectory cannot leak back in.

### A1b. A committed home for the cross-machine dataset

`local.jsonl` is gitignored, and `.gitignore` already anticipates the successor:
*"A curated cross-machine dataset would be committed under a different name."*
Without one, the 5-machine dataset cannot be committed, CI cannot run the gate,
and the README's accuracy number is unreproducible by anyone else.

Extend `fit_cmd::path()` to resolve in order:

1. `ZC_CALIBRATION` — unchanged, keeps tests and validation runs isolated.
2. `data/calibration/gate.jsonl` — the committed cross-machine dataset, when present.
3. `data/calibration/local.jsonl` — whatever this machine produced.

Three lines, no new concept, and it retires the env-var ceremony that would
otherwise be required on every gate invocation. Merging a user's local records
*into* the shipped dataset is a `zc share` concern (Phase C), not this change.

**Verify:** `zc gate` reports 1 of 5 machines and **1** of 2 bare-metal; a unit
test asserts the three-step precedence.

### A2. Release pipeline

New `.github/workflows/release.yml`, triggered on `push: tags` matching `v*`.

Targets (5):

| Target | Runner |
|---|---|
| `x86_64-unknown-linux-musl` | ubuntu-latest |
| `aarch64-unknown-linux-musl` | ubuntu-latest (cross-linker) |
| `x86_64-apple-darwin` | macos-latest |
| `aarch64-apple-darwin` | macos-latest |
| `x86_64-pc-windows-msvc` | windows-latest |

Each job builds `--release`, strips where applicable, emits the binary and a
`.sha256` alongside it, and uploads both as release assets.

Deliberately deferred: release-please, Homebrew tap, winget/scoop, code signing.
Hand-tagging avoids release-please's `GITHUB_TOKEN`-tagging problem (tags pushed
with that token do not trigger downstream workflows) entirely, and SmartScreen
matters at scale, not at first announcement.

`./check.sh` must be green before any tag is pushed.

**Verify:** tag `v0.1.0-rc1`; all five assets appear; on each reachable platform
the downloaded binary runs `zc check --json | python3 -m json.tool` clean.

### A3. Field campaign — 5 machines

Target spread, per PLAN.md Phase 0: 8 GB Windows, Apple Silicon, an old Intel
Mac, a Linux desktop, and something DRAM-less.

Per-machine runbook (documented as `docs/gate-runbook.md` so it can be handed to
someone else):

1. Download the matching release binary and verify its `.sha256`.
2. `zc doctor > doctor-<label>.md` — **before** installing anything, so the probe
   sees the machine as a user would.
3. Install one calibration-grade runtime (ollama, llama.cpp or LM Studio).
4. Pull `qwen3:1.7b` (~1.4 GB) as the common anchor across all machines, plus
   whatever that machine already has locally.
5. `zc verify` once per available model.
6. Carry back `local.jsonl` and the doctor bundle.

The doctor bundles are the campaign's second deliverable and are nearly free.
`VERIFICATION.md` marks the Linux and Windows probe paths as never-executed,
`cpu.rs:291`'s `GetLogicalProcessorInformationEx` is flagged `UNVALIDATED`, and
`FILE_FLAG_NO_BUFFERING` may serialise concurrent reads. This campaign is the
only planned opportunity to exercise those on inspectable hardware.

Records are appended into the committed `data/calibration/gate.jsonl` (A1b), one
line per run. Per-machine provenance is already carried by the `hw` fingerprint,
which is what `zc gate` groups on, so no per-machine file split is needed.

**Verify:** `zc gate` reports ≥5 machines and ≥2 bare-metal.

### A4. Fit, then gate

`zc fit` recomputes coefficients from the merged records. `zc gate` is then the
exit test, and it has two halves:

- **Median error < 25%** — the stated gate.
- **`within_range`** — the published promise is a range, and the only measurement
  on record fell outside it. The 1.645σ coverage factor and the 0.40 prior floor
  were written to fix exactly this and have never been tested against a real
  measurement.

If the median passes but `within_range` stays low, the midpoint is right and the
published width is wrong; that is a coverage-factor and confidence-tier problem,
handled as a physics change, not by narrowing to fit.

Stop-loss, per PLAN.md: if neither converges across 5 real machines, the product
does not work, and week 2 is when that was designed to surface.

**Phase A exits when `zc gate` exits 0.**

---

## Phase B — Launch surface

### B1. README

Does not exist today. Part 1 of the parity plan is its first draft and is already
written and evidenced.

- Opens with the measured gate figure: machine count, median error, date.
- Real terminal output above the fold.
- The measure-don't-assume rule stated plainly — it is the differentiator, and
  every competing tool prints a number for everything.
- Honest positioning against llmfit: narrower, measured, auditable.

Written **after** Phase A, so the accuracy number is real rather than a placeholder.

### B2. `install.sh`

`curl | sh` installer:

- Resolves the latest tag by following the `/releases/latest` redirect rather
  than the API, which avoids the 60 req/h unauthenticated limit.
- Verifies the downloaded `.sha256`.
- Installs to `/usr/local/bin`, falling back to `~/.local/bin`.
- Reads any sudo password from `/dev/tty`, so it works under a pipe.

**Verify:** run on a clean machine that has never had `zc`; then run again to
confirm it is idempotent.

Deferred: a scheduled workflow republishing the gate number. It needs merged
community records, which needs `zc share`. Until then the README carries a static
number and the date it was measured.

---

## Phase C — Post-launch, demand-ordered

1. `zc share` (§2.8) — the only mechanism for Phase 1's <15% / 300-calibration
   target.
2. `zc plan` (§2.5) and live HF lookup (§2.14). §2.14 remains blocked on an open
   decision: quantised byte counts live in a separate GGUF repo, so `zc check
   <hf-repo-id>` can either require a second argument, predict memory only, or
   compute `params × bits-per-weight` and label it an estimate. The last would let
   a non-measured number into a prediction for the first time. Nothing before
   Phase C depends on it, so it stays parked.
3. `zc serve` REST + MCP (§2.10), `zc-tui` (§2.9), web result cards (§2.11).
4. §2.6 remainder: fit-score falloff, `--use-case`, capability metadata.

## Explicitly not built

- **CSV renderer** (named in §2.3). Nothing has asked for it.
- **Homebrew / winget / code signing** in Phase B. Scale problems, not
  first-announcement problems.
- **Docker image.** A hardware prober inside a container measures the container.
- **Auto-tuned coefficients.** Restated because a red gate is exactly when the
  temptation appears.

## Testing

- `./check.sh` green throughout: `cargo test --workspace`, clippy `-D warnings`,
  `cargo check` against all three targets.
- Release assets smoke-tested per platform with `zc check --json` piped through a
  JSON parser.
- `install.sh` run twice on a clean machine.
- No new prediction logic is introduced by this spec, so no new physics tests are
  required. Any change arising from A4's `within_range` finding gets its own
  hand-computed test in the existing house style.
