# The share loop — design

`PLAN.md` Phase 2, parity-plan §2.8. Turns a user who ran `zc verify` into a
contributor to the dataset the predictions are fitted from, without `zc` ever
opening a socket.

## Problem

The Phase 0 gate is blocked on `MIN_BARE_METAL: 1 of 2` — a second physical
machine. No code closes that; only hardware does. The general form of the
problem is worse than the specific one: the moat is "real measured tok/s across
thousands of low-end machines", and there is currently no path from a stranger's
machine into `data/calibration/`. `zc verify` already prints `zc share to
submit` for a command that does not exist.

Timing is load-bearing. Launch attention arrives once. If the share loop is not
live when it does, that traffic produces zero records and the dataset never
starts compounding. The loop has to ship *before* the launch, not after it.

## Decisions taken

1. **No credentials, no egress from `zc`.** The record travels in a URL that the
   *browser* opens, not in a request `zc` makes. `zc check`, `verify`, `fit`,
   `gate`, `doctor` and now `share` make zero outbound connections, and the
   README's privacy claim stays literally true rather than nearly true.
   Rejected: §2.8's GitHub device flow. It is lower friction for a repeat
   contributor and it is what llmfit does, but it costs a registered OAuth App,
   a token cached on user disk, and it makes `zc` a credential-holding program.
   Revisit only if submission volume shows the browser step is the bottleneck.
2. **Two provenance tiers, both published.** `data/calibration/gate.jsonl` is
   maintainer-provenance and backs the headline accuracy claim.
   `data/calibration/community/` holds merged submissions. Both feed `zc fit`,
   so a contributor improves everyone's predictions the day their PR merges;
   only the curated tier moves the headline number and the exit code.
   Rejected: one tier (an unverifiable record moves the published claim), and
   automatic outlier quarantine (it would have hidden the 410% macOS VM, which
   is kept visible on purpose).
3. **Content-addressed filenames.** `<hw>-<hash8>.jsonl`. Resubmitting the same
   run yields the same filename and an empty diff, which is §2.8's
   "append to the existing PR rather than opening a second" without any PR
   machinery to build.
4. **The validator recomputes what the record claims.** `error_pct` and
   `within_range` are both derivable from fields the record already carries, so
   a fabricated "0% error" submission has to contradict itself to exist.

## Verified before designing

- GitHub docs, `creating-new-files.md`: "If you attempt to create a file in a
  repository you do not have write access to, GitHub will automatically fork the
  project to your personal account and assist in opening a pull request to the
  original repository after you commit your changes." The fork-and-PR half needs
  no code.
- A real 514-byte record makes an 888-byte URL. Requesting it anonymously
  returns 302 to `/login` with `filename` and `value` preserved intact in
  `return_to`, so the parameters survive authentication and reach the editor.
- GitHub returns **414 URI Too Long** past its query limit, so the length guard
  in §3 is a real requirement rather than defensive decoration.

## Standing constraint

A number is measured, derived from measured inputs, or printed as `-`. The share
loop adds the first path by which numbers this project did not measure enter the
dataset, which is exactly why §4 recomputes every derivable field and §2 keeps
the headline claim on the tier whose provenance is known.

---

## 1. `zc share`

    zc share [--record FILE] [--print]

Reads the last line of `data/calibration/local.jsonl`, or of `--record FILE`.
Reads *the last line* because `zc verify` appends and the most recent run is the
one the user just watched.

Output, in order:

1. The record, one field per line, formatted — not the raw JSON. This is the
   `PLAN.md` C1 promise ("print exactly what's collected before sending")
   discharged at the only moment it matters.
2. The line stating what is **not** in it: no hostname, username, serial, MAC,
   IP or file path.
3. The destination filename.
4. The URL.

Then, on a TTY, `open in browser? [y/N]`. Anything but `y` exits 0 having done
nothing. Not a TTY, or `--print`: no prompt, the URL is printed and that is all,
so the command composes in a pipeline without ever surprising anyone.

`open` on macOS, `xdg-open` on Linux, `cmd /c start` on Windows. If the opener
is missing or fails, the URL is already on stdout and the user pastes it.

**Failure modes, all of which print and exit non-zero rather than panicking:**
no record file, empty record file, unparseable last line, and a URL exceeding
6 KB — under GitHub's 414 threshold with room to spare, and the fallback is to
print the filename and the record for manual creation.

**Exit codes.** 0 on success or a declined prompt, 1 on any failure above.

## 2. Where records live, and what reads them

    data/calibration/
      gate.jsonl          curated. maintainer provenance. backs the headline.
      community/          merged submissions, one record per file.
      local.jsonl         this machine's runs. gitignored. unchanged.
      archive/            retired records. read by nothing. unchanged.

`fit_cmd::path()` grows a sibling, `fit_cmd::sources()`, returning the curated
file plus every `community/*.jsonl`. `Fit::load` consumes the concatenation, so
predictions and coefficients see both tiers.

`zc gate` computes `Gate::from_records` **twice** — curated alone, then curated
plus community — and prints both, labelled. The pass/fail decision and the
process exit code follow the curated number only. `Gate` is already a pure
function of a record slice, so this is a second call, not a second code path.

Promotion from community to curated is `git mv`. Deliberately no tooling: it
should cost a visible commit with a human's name on it.

## 3. URL construction

    https://github.com/DEEPESH-845/ZeroCloud/new/main
      ?filename=data/calibration/community/<hw>-<hash8>.jsonl
      &value=<the record line>

Both parameters percent-encoded with an unreserved set of `A-Za-z0-9-_.~`;
everything else, including `/`, is escaped. Hand-rolled, ~15 lines, because the
workspace takes no dependencies.

`hash8` is the low 32 bits of FNV-1a-64 over the exact record line, lowercase
hex. Not a security property — the validator recomputes it, and its job is
idempotent naming, not authentication.

The `value` parameter is longstanding but undocumented GitHub behaviour. If it
ever stops working the contributor lands in the editor with the right filename
and an empty body, having already been shown the record on their terminal. The
loop degrades to copy-paste rather than breaking.

## 4. `scripts/validate_calibration.py`

Python 3 stdlib only, matching `scripts/ingest_hf.py`. Runs over the **whole**
`data/calibration/` tree on every invocation, not only changed files: repo-wide
integrity is the invariant, and a PR that corrupts an untouched file is exactly
what a changed-files-only validator misses.

Per file in `community/`:

- filename matches `^[0-9a-f]{16}-[0-9a-f]{8}\.jsonl$`
- exactly one line, at most 4 KiB
- parses as a JSON object; no unknown top-level keys, so nothing can be smuggled
  past the schema into a future reader
- the filename's `hw` segment equals the record's `hw`, and its `hash8` segment
  equals FNV-1a-64 of the line

Per record, in `community/` and in `gate.jsonl` alike:

- required fields present, `error_pct` among them — `gate.rs` counts a record
  without one as `skipped`, and the validator refuses it at the door
- `virt` ∈ {`none`, `hypervisor`, `wsl2`, `container`}
- `implied_eta` finite and in `(0, 1.5]` — `fit.rs` discards the rest, so
  enforce it at submission rather than silently dropping it later
- `0 < predicted_lo ≤ predicted_hi`, `actual_decode_tok_s > 0`
- **`error_pct` recomputed** from the published midpoint against the actual, and
  required to agree within 0.15 (the record carries one decimal place)
- **`within_range` recomputed** from `predicted_lo`/`predicted_hi` against the
  actual, and required to agree exactly
- plausibility bounds a schema cannot express: `1 ≤ ram_bw_gbs ≤ 2000`,
  `0 < gflops ≤ 1e6`, `1 ≤ threads ≤ 1024`, `0 < ctx ≤ 10_000_000`

Every failure is reported with file, line and reason; exit 1 if any. `--self-test`
runs the fixtures in §6 and touches no repository file.

## 5. CI

`.github/workflows/calibration-prs.yml`: on `pull_request` touching
`data/calibration/**`, checkout and run the validator. The check status is the
whole signal — no bot comment, no auto-merge, no auto-approval. Validation is
the only automated gate and a human merges, which is the part of llmfit's design
worth copying exactly.

The same script runs in `ci.yml` and in `check.sh`, so `main` cannot drift into
an invalid dataset between PRs.

## 6. Testing

House style: assert a hand-computed quantity, or lock in a specific bug.

Rust, in `share.rs`:
- the filename is content-addressed — a record and its byte-for-byte copy give
  the same name; flipping one digit of `actual_decode_tok_s` gives a different one
- percent-encoding round-trips a record containing `&`, `"`, `+`, `/` and a
  non-ASCII model name, decoded back with a hand-written decoder in the test
- what is printed equals what is encoded: the shown record and the `value`
  parameter come from one string, and the test proves they cannot diverge
- a missing record file, an empty one and a garbage last line each produce a
  message and exit 1

Python, under `--self-test`:
- a record whose `error_pct` disagrees with its own `predicted_*`/`actual_*` fails
- a record whose `within_range` disagrees with its own bounds fails
- the corresponding correct record passes
- a two-line file fails; a misnamed file fails; a file with an extra top-level
  key fails

## 7. Embedding

`crates/zc-model/build.rs` already globs `data/models/*.json`. It grows a second
glob over the curated file plus `community/*.jsonl`, emitting them sorted for a
reproducible build, and `fit.rs`'s hardcoded `include_str!` of `gate.jsonl` is
replaced by the generated constant. An installed user keeps predicting from the
full merged dataset rather than from priors.

## Explicitly not built

Named so nobody adds them mid-plan: GitHub device flow, token caching, forking
or opening PRs from the CLI, appending to an existing PR by API, web result
cards (§2.11), `zc serve` (§2.10), the TUI (§2.9), promotion tooling between
tiers, and any submission path that uploads without the user seeing the payload
first.
