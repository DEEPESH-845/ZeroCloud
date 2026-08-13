# Week 1 — Verification Report

Machine: Apple M5, 4P+6E, 16 GiB unified, APFS on internal NVMe, macOS 26.5.2, bare metal.

## Verified accurate

| Measurement | Ours | Reference | Method |
|---|---|---|---|
| RAM total | 16.00 GiB | 16.00 GiB | `sysctl hw.memsize` |
| Physical cores | 10 (4P/6E) | 10 (4P/6E) | `hw.perflevel0/1.physicalcpu` |
| Cache line | 128 B | 128 B | `hw.cachelinesize` |
| ISA features | dotprod, i8mm | present | `hw.optional.arm.FEAT_*` |
| Virtualisation | none | none | `kern.hv_vmm_present` = 0 |
| Mount point | `/System/Volumes/Data` | correct APFS firmlink | `statfs` |
| RAM read bandwidth | 132 GB/s | ~153 GB/s theoretical → **86%** | plausible for a read benchmark |
| Compute | 118 GFLOPS/core | ~147 theoretical → **80%** | 4 pipes × 4 lanes × 2 flops @ ~4.6 GHz |
| Disk cache bypass | confirmed | — | see below |

**Cache-bypass proof.** RAM reads measure 132 GB/s. Disk reads measure 5.0 GB/s. If
`F_NOCACHE` had failed, repeated reads of a 512 MB file would be served from page
cache at RAM speed. A 26× gap proves we are reading media. No assertion required —
the two measurements check each other.

**KV cache math.** Unit-tested against hand-computed values, exact equality:

- Llama-3.1-8B: `2 × 32 × 8 × 128 × 2` = **131,072 B/token** = exactly 128 KiB, and
  exactly 1 GiB at 8K context.
- Llama-3.3-70B: `2 × 80 × 8 × 128 × 2` = **327,680 B/token**, and exactly 10 GiB at
  32K context — larger than the entire target machine.
- MLA (DeepSeek-V3): `61 × (512+64) × 2` = **70,272 B/token** for a 671B model, less
  than a quarter of the 70B GQA figure. An order of magnitude less KV for ten times
  the parameters.
- SWA (Gemma-3): holds 27% of what full attention would at 8K context.
- Hybrid SSM: grows with attention layers only.

## Bugs found by verification

Five real defects, three of which would have produced badly wrong predictions.

1. **FMA benchmark was latency-bound, not throughput-bound.** 16 lanes = 4 vector
   chains; with ~4-cycle FMA latency and 4 pipes we could only retire one per cycle.
   Caught by dividing the result by clock speed and finding 7.8 flops/cycle against a
   theoretical 32. Fix: 64 lanes = 16 chains. **36 → 118 GFLOPS (3.3×).**

2. **Budget computed from instantaneous free memory.** Reported 1.98 GiB because a
   browser was open, so every model showed "won't fit". Fix: predict against
   `potential_budget` (idle machine), report `current_budget` alongside.

3. **Bandwidth selection was backend-blind.** Predicted Metal inference from a
   CPU-core bandwidth measurement. On unified memory the GPU saturates the memory
   controller far better than the CPU cluster (132 vs 85 GB/s here). Fix: `Backend`
   enum selects the correct figure.

4. **Benchmark-file heuristic too loose.** A filename-contains-`-` rule matched an
   `.mp4`. Fix: `sha256-` prefix or known extension; separate "any large file"
   fallback with an honest label.

5. **A test asserted a guessed threshold.** SWA growth ratio was 1.615; the test
   demanded < 1.6. The arithmetic was right and the test was wrong. Replaced with the
   property that actually matters (SWA vs full attention at fixed context).

## NOT verified — this is the Week 1 gate

**`DECODE_EFFICIENCY = 0.70` is a prior, not a measurement.** No prediction has been
compared against a real inference run. Until `zc verify` exists and runs against a
real model, every tok/s number in the output is unvalidated.

There is already a signal worth chasing: the MoE prediction (Qwen3-30B-A3B IQ3_XXS,
~47 tok/s midpoint) sits close to commonly reported figures, while the dense
predictions (Llama-3.1-8B Q4_K_M, ~16.5 midpoint) look roughly 30% low. If that holds
up under measurement, `η` needs more spread across quant families than the current
constants allow — K-quants nearer 0.85, I-quants nearer 0.50. **This is a hypothesis
to test, not a finding, and the constants must not be tuned before the measurement.**

## Known-wrong, documented in code

| Gap | Impact | Fix |
|---|---|---|
| TTFT from f32 CPU FMA | **10–40× understated**; real prefill uses GPU or int8 kernels | measure f16/int8 FMA + Metal/CUDA probe in `zc-bench` |
| Linux/Windows paths never executed | unknown | run on real hardware; Windows `GetLogicalProcessorInformationEx` walk is marked `UNVALIDATED` |
| `firmware_reserved` unimplemented off macOS | iGPU carve-out (1–2 GiB) uncounted — matters most on the target hardware | SMBIOS type-17 sum vs OS-visible total |
| Discrete GPU unmodelled | VRAM tier ignored | `Backend::Discrete` |
| `compute_buffer_bytes` multiplier is a guess | context math off by ~100–300 MB | calibrate |

## `zc verify` — built, and what it does

The command that turns constants into measurements. It reads Ollama's own
instrumentation (`prompt_eval_duration` / `eval_duration`) rather than timing from
outside, so no HTTP or process-startup overhead lands in the numbers, and it gets the
prefill/decode split for free.

It builds the ModelSpec from `/api/show` GGUF metadata, so it works on **any** model
the user has installed rather than only catalog entries — architecture, layer count,
KV head count, sliding window and MoE expert geometry are all read, never inferred.

The core operation is inverting the prediction model:

```text
tok_s        = eta / raw_seconds_per_token      (predict)
implied_eta  = actual_tok_s * raw_seconds_per_token   (verify)
```

Accumulating `implied_eta` across models, quant families and machines *is* the
calibration dataset.

### Bugs found while building it

6. **`split("},{")` for JSON array elements.** Worked against compact JSON, returned
   one giant chunk the moment any whitespace appeared between elements — so a proxy
   or a future Ollama that pretty-printed would have silently reported one model.
   Caught by a unit test using indented fixture JSON. Replaced with a depth- and
   string-aware `array_objects`.

7. **`zc verify` benchmarked for 20 s before checking whether a runtime existed.**
   Users without Ollama waited, then got told to install Ollama. Moved the check
   ahead of the probe: 20 s → 0.9 s.

8. **Hardcoded endpoint made the happy path untestable.** Replaced with an
   `Endpoint` type honouring `OLLAMA_HOST` — which real users need anyway — and
   backed by a mock-server integration test that exercises TCP, HTTP framing, JSON
   extraction and rate computation without a runtime installed.

### Correctness measures worth noting

- **Prompt nonce per run.** Ollama caches KV for repeated prompt prefixes. An
  identical prompt would report near-zero prefill time and silently corrupt the
  calibration, so warm-up and measurement use different nonces.
- **Warm-up run first**, so model load time is not charged to the first tokens.
- **Smallest model is the default target** — a first run that takes ten minutes is a
  first run nobody finishes.
- **Privacy.** Records contain a salted FNV hash of a hardware *profile*, plus OS,
  backend and benchmark figures. No hostname, username, serial, MAC, IP or path.
  Written to `data/calibration/local.jsonl` and nothing leaves the machine without an
  explicit `zc share`.

## State

```
5 crates · 3,974 LOC · 37 tests passing · 0 clippy warnings · 500 KB release binary
zero runtime dependencies beyond libc / windows-sys
```

| Crate | Tests | Role |
|---|---|---|
| `zc-probe` | — | memory, CPU, environment, storage detection |
| `zc-bench` | — | RAM bandwidth, uncached disk, FMA throughput |
| `zc-model` | 8 | 4 attention architectures, KV math, budget, prediction |
| `zc-runtime` | 29 | Ollama client, JSON, HTTP, calibration inversion |
| `zc-cli` | — | `zc check` / `zc verify` |

## Next

1. **Install Ollama and run `zc verify` across ≥3 models × 2 quant families.** Every
   tok/s number remains unvalidated until this happens.
2. Refit `η` from `data/calibration/local.jsonl`. Do not tune it by hand first.
3. Fix the prefill measurement (f16/int8 FMA + Metal/CUDA probe).
4. Run on Windows and Linux boxes.
5. `zc share` — submit a record via a GitHub Action PR.

**Gate: median decode prediction error < 25% across ≥5 machines.** Not yet met. The
instrument that measures it now exists; the measurement does not.

---

# Phase 1 — Catalog & Calibration

## Data-driven catalog

The catalog moved from hardcoded Rust to `data/models/*.json`, embedded at build
time by `build.rs`. This is the community contribution surface: adding a model is a
one-file PR needing no Rust and conflicting with nobody. A directory on disk
(`ZC_DATA_DIR`) overrides the embedded set so contributors can iterate without
recompiling, and embedding keeps the binary self-contained and offline-capable.

**The tests validate the data, not just the parser.** Each catalog entry is checked
against a hand-computed KV figure, so a wrong `n_kv_heads` or `head_dim` in a JSON
file fails CI rather than silently corrupting every prediction for that model:

| Model | Asserted | Derivation |
|---|---|---|
| llama-3.1-8b | 131,072 B/token | `2 × 32 × 8 × 128 × 2` |
| llama-3.3-70b | 327,680 B/token, exactly 10 GiB at 32K | `2 × 80 × 8 × 128 × 2` |
| qwen3-4b | 147,456 B/token | `2 × 36 × 8 × 128 × 2` |
| deepseek-v3 (MLA) | 70,272 B/token | `61 × (512+64) × 2`, no K/V factor of 2 |
| qwen3-30b-a3b | 3.0–3.7B active | must reproduce the advertised "A3B" |

Entries with an unknown attention kind are **rejected**, not guessed at — predicting
with the wrong KV formula is worse than showing nothing.

## Fitting eta from calibration records

`zc fit` turns `implied_eta` measurements into the coefficients `predict` uses.

**Median, not mean.** A run during thermal throttling or a background build is
arbitrarily slow, and nothing makes a run arbitrarily *fast* — the error distribution
has a long left tail only. A mean gets dragged into it by one bad sample.

**MAD, not standard deviation.** One outlier inflates a standard deviation without
bound, which would widen the published range for every user.

**Confidence tiers gate how narrow the range may get.** Thresholds are deliberately
conservative: a confident wrong number is what kills a prediction tool, so staying
wide too long costs far less than narrowing too early.

| Samples | Confidence | Minimum range half-width |
|---|---|---|
| 30+ | high | ±8% |
| 8–29 | medium | ±15% |
| 1–7 | low | ±25% |
| 0 | prior | ±30% |

Observed spread overrides the floor when wider — many samples that disagree is not
confidence. Records with `eta > 1.5`, ≤ 0, or non-finite are discarded: those imply a
run beat the machine's own measured bandwidth, which is impossible.

## Validated end-to-end with a synthetic dataset

45 synthetic records (not measurements — a pipeline test), including one deliberately
throttled outlier at η=0.11:

```
bucket                      runs       eta    spread  confidence
Metal/k_quant                 31     0.840        8%  high
Metal/i_quant                 10     0.505       15%  medium
Cpu/legacy                     4     0.600       25%  low
```

The outlier moved the fit not at all (mean would have been 0.826). Predictions with
and without the dataset, same machine:

| | priors only | calibrated |
|---|---|---|
| llama-3.1-8b Q4_K_M | 10.1–18.8 tok/s, `prior` | 20.5–24.1 tok/s, `high` |
| range width | ±30% | ±8% |

The range narrowed 3.75× and the midpoint moved with the evidence. That is the loop
working.

## Bugs found in this phase

9. **Parent-bucket family bias.** When a quant family had never been measured on a
   backend, the fallback used the pooled median across *all* families on it — biased
   toward whichever family had the most samples. A parent built mostly from i-quants
   would badly understate a legacy quant. Fixed by transferring the *ratio*
   (`parent_eta × family_prior / parent_mean_prior`) rather than the raw value.
   Caught by inspecting the validation output, not by a test.

10. **`cargo test | grep "test result"` hid a compile failure.** Filtering a build's
    output down to success patterns made two broken `Prediction` literals invisible
    for one cycle. Never filter a build to only the lines you hope to see.

## State

```
5 crates · 4,835 LOC · 57 tests passing · 0 clippy warnings · 566 KB release binary
zero runtime dependencies beyond libc / windows-sys
```

Commands: `zc check` · `zc verify [MODEL]` · `zc fit`

## Still open

The Week 1 gate is unchanged: **no real inference run has been measured.** The
synthetic validation proves the machinery is correct; it proves nothing about the
coefficients. `DECODE_EFFICIENCY = 0.70` remains a prior until `zc verify` runs
against a live model.

---

# Phase 2 — The Prefill Fix

The largest known-wrong number: TTFT was understated by 10–40×. Fixed by finding out
*why*, rather than by adjusting a constant.

## What was actually wrong

Two independent errors, not one:

1. **Wrong primitive.** llama.cpp's Q4_K prefill runs `vec_dot_q4_K_q8_K` — an int8
   dot product. We measured f32 FMA.
2. **Wrong device.** Real prefill runs on the GPU wherever one exists. We measured
   the CPU.

`PREFILL_EFFICIENCY = 0.35` was silently absorbing both, plus the legitimate
GEMM-vs-peak-FMA gap. Three unrelated factors in one constant, spanning more than an
order of magnitude.

## Measured: stable Rust cannot reach the int8 path

Rather than assume, three kernels were benchmarked (M5, 4 threads):

| int8 dot kernel | Throughput |
|---|---|
| separate-lane widening MAC | **363 GOPS** |
| sdot-shaped (4 products → 1 accumulator) | 108 GOPS |
| explicit `vdotq_s32` intrinsic | **will not compile** |

Three findings:

- `vdotq_s32` is gated behind the unstable `stdarch_neon_dotprod` feature.
- LLVM will not auto-lower to SDOT even with `-C target-feature=+dotprod`.
- The sdot *shape* is 3.4× **slower** in portable Rust — its inner reduction forms a
  dependency chain, the same latency-versus-throughput trap that cost 3.3× in the f32
  benchmark earlier.

So no portable-Rust benchmark can measure achievable quantised prefill. Real runtimes
reach it through intrinsics and go several times faster than anything above.

## The fix: measure it, or say you do not know

Prefill is now fitted from real runs, exactly like decode's eta:

```text
predict:  prefill_tok_s         = gflops * prefill_scale / (2 * active_params)
verify:   implied_prefill_scale = actual_prefill_tok_s * 2 * active_params / flops
```

`prefill_scale` is deliberately **not** called an efficiency — it routinely exceeds
1.0, because the GPU beats our CPU f32 benchmark. It is a device/path ratio absorbing
GEMM-vs-peak, int8-vs-f32 and GPU-vs-CPU together, and nothing clamps it.

**There is no prior.** `Prediction::prefill_tok_s` and `ttft_s` are `Option<f64>`, and
`fit::prefill_scale` returns `None` for an unmeasured backend. With no measurement the
report prints `-` and explains why. A dash is honest; a derived number would be wrong
by an unknown factor, and a confident wrong number is what kills a prediction tool.

Prefill is grouped by **backend only** — it is dominated by which device runs the
GEMM, not by weight quantisation — and there is no cross-backend fallback, because a
CPU measurement says nothing about a GPU.

## Validated

| | before | after (no data) | after (measured) |
|---|---|---|---|
| llama-3.1-8b Q4_K_M TTFT | 220s | `-` + explanation | **15.2s** |
| qwen3-30b-a3b TTFT | 91s | `-` | **6.3s** |

The MoE case is a good check that the physics is right: a 30B model shows faster TTFT
than an 8B dense one, because only 3.3B parameters are active.

## State

```
5 crates · 5,089 LOC · 61 tests passing · 0 clippy warnings · 582 KB release binary
```

## Known-wrong list, updated

| Gap | Status |
|---|---|
| ~~TTFT from f32 CPU FMA, 10-40x understated~~ | **fixed** — fitted or reported unknown |
| Linux/Windows paths never executed | open |
| `firmware_reserved` unimplemented off macOS | open |
| Discrete GPU unmodelled (`Backend::Discrete`) | open |
| `compute_buffer_bytes` multiplier is a guess | open |
| int8 benchmark cannot reach SDOT | documented; no longer load-bearing |

The Week 1 gate is still open: no real inference run has been measured. Both `eta` and
`prefill_scale` now have the machinery to be measured, and neither has been.

---

# Audit — Phase 0 Readiness

A deliberate look for defects rather than a re-assertion of prior claims. Six found,
all fixed.

## Bugs found and fixed

**11. `max_context` contradicted the streaming model.** It subtracted the *full*
weight size from the budget, so any model larger than RAM returned 0 context and
`WontFit` — while the same function simultaneously computed a streaming decode speed
for that model. The two halves disagreed.

Physically, KV must be resident (written every token, cannot stream) and weights need
not be (read-only, mmap-able). Weights should never *block* context, they trade
against it. Rewritten as an explicit policy: reserve a 2048-token KV floor first, then
let the weight cache take what remains. `WontFit` is now reserved for the genuine case
where not even a minimal context fits.

Effect: `qwen3-30b-a3b Q4_K_M` went from `WontFit` to **4.5–8.4 tok/s at 81%
resident**, and `llama-3.3-70b Q4_K_M` from `WontFit` to **0.1 tok/s** — which
independently reproduces the 0.14 tok/s figure hand-derived at 6 GB/s in the original
spec review, scaled to this machine's measured 4.98 GB/s. The implementation now
agrees with the analysis that motivated the project.

**12. MoE residency was systematically understated.** `resident_fraction` was computed
over *total* bytes and applied to *active* bytes. For MoE the hot core is ~5% of the
file but ~45% of what is read per token, and it is pinned first. Replaced with a
two-tier model: hot core pinned, remainder caching routed experts, each contributing
separately to bytes-from-RAM. Dense models degenerate to the simple cache ratio.

**13. `resident.max(hot)` claimed the hot core was resident when it did not fit.**
Optimistic in the wrong direction. Regression test added.

**14. The `zc verify` prompt was ~40 tokens.** Prefill measured over 40 tokens is
dominated by fixed per-request overhead — far too noisy to fit a coefficient from,
which is exactly what `implied_prefill_scale` needs. Replaced with a ~1200-token
varied, nonce-prefixed prompt.

**15. zc-bench did not compile on Windows at all.** `std::os::unix::fs::FileExt` was
used unconditionally, outside any `cfg`. Windows also had no cache-bypass path, so it
would have reported page-cache speeds as disk speeds. Added a cross-platform
`read_at`/`write_at` shim and `FILE_FLAG_NO_BUFFERING`. Windows is the primary target
market and the code had never been compiled for it.

**16. Missing `windows-sys` feature** (`Win32_Storage_FileSystem`) — only a real
cross-compile surfaces this.

## Dead code removed or surfaced

`int8_gops_nt`, `int8_ratio`, `bench_is_weight` were computed but never displayed;
they now appear in `zc check`. A failed disk measurement previously fell back to a
silent 0.5 GB/s default — it now says so loudly, because streaming speed is predicted
from it. `run_threads` no longer panics if every worker dies.

## Cross-platform status

`./check.sh` runs tests, clippy with `-D warnings`, and `cargo check` against all
three targets. Cross-target checking is not optional: zc-bench was broken on Windows
for days and nothing caught it.

| Target | Status |
|---|---|
| aarch64-apple-darwin | 0 errors, 0 warnings, **runs** |
| x86_64-unknown-linux-gnu | 0 errors, 0 warnings, type-checked only |
| x86_64-pc-windows-msvc | 0 errors, 0 warnings, type-checked only |

Type-checked is not validated. It proves the code compiles, not that `MemAvailable`
parsing, `GetLogicalProcessorInformationEx` walking, or `FILE_FLAG_NO_BUFFERING`
behave correctly.

## State

```
5 crates · 5,400 LOC · 67 tests · 0 clippy warnings · 3 targets clean
```

## What is still missing for Phase 0

Phase 0's gate: **median decode prediction error < 25% across ≥5 machines.**

| Requirement | Status |
|---|---|
| Benchmark harness | done |
| Prediction math, unit-verified | done |
| `zc verify` measurement instrument | done |
| Calibration fitting with confidence tiers | done |
| Compiles on all three platforms | done |
| **Real inference measured, even once** | **not done** |
| **≥5 machines** | **1 of 5** |
| **Windows/Linux behaviour validated** | **not done** |
| **Gate computed** | **not possible yet** |

Blocking work, in order:

1. `brew install ollama`, pull 3 models across 2 quant families, run `zc verify`.
   Until this happens every tok/s number is a prior and the gate is unmeasurable.
2. Run on a Windows 8 GB laptop. Two things there are unvalidated and load-bearing:
   the `EfficiencyClass` core walk, and whether `FILE_FLAG_NO_BUFFERING` without
   `FILE_FLAG_OVERLAPPED` serialises concurrent reads (which would understate
   queue-depth bandwidth).
3. Implement `firmware_reserved` off macOS (SMBIOS type-17 sum vs OS-visible total).
   An uncounted 1–2 GiB iGPU carve-out is worst exactly on the 8 GB machines this
   product exists for.
4. Three more machines: old Intel Mac, Linux desktop, something DRAM-less.
5. Compute the median error and pass or fail the gate honestly.

Items 1–3 are the ones that need code or a package install. Items 4–5 need hardware.
