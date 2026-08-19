# Phase 0 gate runbook

The gate needs **≥5 distinct machines**, **≥2 on bare metal**, and a **median of
per-machine medians below 25%**. It reads `error_pct` from each record rather
than recomputing it, so every error is genuinely out-of-sample: the prediction
was made before that run existed.

Current state: run `zc gate` to see it.

## Where the machines come from

Three of the five are already automated. `.github/workflows/calibrate.yml` runs
`zc verify` on ubuntu-latest, macos-latest and windows-latest and merges the
three records into a `calibration-merged` artifact — it just never commits them.

```sh
gh workflow run calibrate.yml
gh run watch
gh run download -n calibration-merged -D /tmp/ci-records
cat /tmp/ci-records/all.jsonl >> crates/zc-model/data/calibration/gate.jsonl
```

Those runners are VMs. They count as machines and they can **never** satisfy
`MIN_BARE_METAL`, which is exactly why that floor exists — hypervisors run 10–30%
below real hardware, so cloud runners alone must not be able to turn the gate
green.

That leaves **at least one more bare-metal machine as mandatory**. Run the wider
physical spread anyway where the hardware is available: a gate that squeaks green
on three cloud VMs and two laptops is weaker evidence than five real machines,
and the extra records cost only time.

Target spread, per `PLAN.md` Phase 0 — the point is coverage of failure modes,
not five of the same laptop:

| Slot | Machine | Why it is on the list |
|---|---|---|
| 1 | Apple Silicon Mac | unified memory, Metal backend |
| 2 | 8 GB Windows laptop | the primary target market; the only test of the WMI + registry GPU path |
| 3 | Old Intel Mac | discrete or integrated Intel, non-unified memory |
| 4 | Linux desktop | the sysfs / lspci GPU path |
| 5 | Anything DRAM-less or single-channel | the failure mode a lookup table cannot see |

## Per machine

1. **Install.** macOS/Linux:

       curl -fsSL https://raw.githubusercontent.com/DEEPESH-845/ZeroCloud/main/install.sh | sh

   Windows: download `zc-x86_64-pc-windows-msvc.exe` from the releases page and
   verify it against the `.sha256` beside it.

2. **Doctor first, before installing anything else.**

       zc doctor > doctor-<label>.md

   Do this before the runtime is installed, so the probe sees the machine as a
   new user's would. These bundles are the campaign's second deliverable:
   `VERIFICATION.md` marks the Linux and Windows probe paths as never-executed,
   `cpu.rs`'s `GetLogicalProcessorInformationEx` is flagged `UNVALIDATED`, and
   `FILE_FLAG_NO_BUFFERING` may serialise concurrent reads. This is the only
   planned chance to exercise them on hardware someone can inspect.

   The bundle carries no hostname, username, serial, MAC or IP, and rewrites
   paths to `~`.

3. **Install one calibration-grade runtime** — Ollama, llama.cpp
   (`llama-server`) or LM Studio. vLLM, MLX and Docker Model Runner are detected
   but refused: their APIs report no prefill/decode split, so a rate measured
   through them would include HTTP and scheduling time.

4. **Pull the anchor model.**

       ollama pull qwen3:1.7b

   `qwen3:1.7b` (~1.4 GB) is a useful common anchor: one model measured on every
   machine makes cross-machine comparison apples-to-apples and exposes any error
   specific to one model.

   It is a convenience, not a requirement. `zc fit` buckets on
   `backend × quant_family`, not on model, so **measuring different models on
   different machines adds evidence rather than fragmenting it** — that is also
   why `calibrate.yml` defaults to the smaller `llama3.2:1b` on CI and is left
   that way. Measure whatever each machine already has: buckets need 8 runs for
   medium confidence and 30 for high, and every extra run counts.

5. **Measure.**

       zc verify qwen3:1.7b

   Repeat per model. Each run appends one line to `local.jsonl` in your data
   directory and prints the full path. Inside a checkout that is
   `crates/zc-model/data/calibration/local.jsonl`; outside one it is the platform's per-user
   data directory, so it does not matter which directory you run from.

6. **Carry back** that `local.jsonl` and the doctor bundle.

   That is the maintainer's route, and it is the one that puts a record in the
   curated tier. Anyone else runs `zc share` instead: it opens a prefilled
   GitHub editor and the record lands in `crates/zc-model/data/calibration/community/`, where it
   moves every coefficient but not the published accuracy figure. The
   distinction is provenance, not quality — promoting a community record into
   `gate.jsonl` is a `git mv` a maintainer makes deliberately.

## Merging

Append each machine's lines to the committed dataset and drop the doctor bundle
in place:

    cat /path/from/machine/local.jsonl >> crates/zc-model/data/calibration/gate.jsonl
    cp /path/from/machine/doctor-<label>.md docs/doctor-bundles/

Per-machine provenance is already carried by the `hw` fingerprint, which is what
`zc gate` groups on, so no per-machine file split is needed.

## Closing

    zc fit    # what the dataset now says, and how much evidence backs it
    zc gate   # exits non-zero until it passes

`zc fit` moves coefficients from records. **Never edit a coefficient by hand**,
including when the gate is red and there is a hypothesis about which direction it
should move. That rule is what makes the published number mean anything.

## Reading the result

Two numbers matter, and the second is the one that can hurt:

- **median error < 25%** — the stated gate.
- **`within_range`** — the published promise is a range, not a point.

**Answered, 2026-08-19.** The early records missed their range in the same
direction — we predicted *slower* than reality, implied eta 0.859 then 0.922
against an assumed 0.616 — and `zc fit` moved Metal/k_quant to 0.922 from that
one record. Two fresh runs then tested that fitted number out of sample and both
landed inside the range: 0.888 and 0.863. So the width mechanism is sound, and a
`within_range` figure that includes predictions made against the shipped prior is
lagging rather than damning. No coverage factor was changed.

The bucket is still `low` confidence, and now for the right reason: three runs,
**one machine**. The tier counts distinct machines, so repeat runs here sharpen
eta and can never narrow the published range on their own.

If the median passes but `within_range` stays low, the midpoint is right and the
published *width* is wrong. That is a coverage-factor and confidence-tier problem
in `crates/zc-model/src/fit.rs`, and it is fixed by widening honestly — never by
narrowing to fit.

**Stop-loss**, per `PLAN.md`: if neither number converges across five real
machines, the product does not work, and finding that out in week two is what the
gate was designed to do.
