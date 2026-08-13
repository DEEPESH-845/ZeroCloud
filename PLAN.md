# ZeroCloud — Execution Plan
## Product: hardware→model compatibility oracle for local AI ("Can You Run It" for LLMs)

---

## 0. The one architectural decision that de-risks everything

**Do not predict from a hardware spec database. Measure three numbers on the user's actual machine, then predict analytically.**

A spec-lookup approach ("i5-1135G7 → 3200 MT/s dual channel → 51 GB/s") is wrong constantly, and every way it's wrong is concentrated in our target market:

- single-channel RAM (half bandwidth, invisible, *extremely* common in budget laptops)
- DRAM-less SSDs (4× worse random read than the spec sheet)
- iGPU stealing 1–2 GB of system RAM
- thermal throttling already in progress
- WSL2 / VM / container memory caps
- Windows Defender destroying disk throughput
- BitLocker/FileVault read overhead

A 20-second micro-benchmark catches **all of these automatically**. The community hardware database then becomes a *validation set*, not the prediction mechanism — which is a much safer place for it to be.

### The three measurements

| # | Measures | Method | Why it's hard to get right |
|---|---|---|---|
| 1 | **RAM bandwidth** (GB/s) | STREAM-triad, multi-threaded, working set > L3×4 | Must exceed cache or you measure L3. Thread count matters. |
| 2 | **Disk read** (GB/s @ QD1, QD32) | O_DIRECT / F_NOCACHE, random 128 KB reads, on **the model directory's** volume | Page cache will report 10 GB/s if you don't bypass it |
| 3 | **Compute** (GFLOPS) | Small tiled sgemm/qgemm, warm, 3 s | Must use the same ISA path the runtime will (AVX2/AVX512/NEON) |

Plus a 30-second sustained pass to detect thermal throttle (bandwidth decay curve).

### The prediction math

```
# Memory budget
igpu_reserved  = detected shared-VRAM carve-out          (0 on Apple Silicon)
os_headroom    = max(1.5 GB, 0.15 × total_ram)           (tuned per OS)
usable         = available_ram − os_headroom − igpu_reserved

# KV cache — the thing that actually OOMs people mid-conversation
kv_per_token   = n_layers × 2 × n_kv_heads × head_dim × bytes_per_elem
compute_bufs   = f(batch, hidden, n_layers)              ≈ 200–600 MB for llama.cpp
max_context    = (usable − model_bytes − compute_bufs) / kv_per_token

# Decode speed — memory-bound
active_bytes   = dense ? model_bytes : (shared + k×expert_bytes)
f              = fraction of active weights resident in RAM
t_token        = f·active_bytes/BW_ram + (1−f)·active_bytes/BW_disk
tok_s          = η(quant, backend, threads) / t_token

# Prefill — compute-bound, completely different formula
tok_s_prefill  ≈ GFLOPS_measured / (2 × active_params)
TTFT           ≈ prompt_len / tok_s_prefill
```

`η` (efficiency, 0.55–0.85) is **not derivable** — it comes from the calibration dataset. That's the moat.

### Why this becomes defensible

The closed loop turns the biggest risk into the biggest asset:

```
predict → user runs the model → we measure actual → user submits (opt-in) → η improves
```

Nobody else has real measured tok/s across thousands of low-end machines. After 5,000 submissions the predictions are better than anyone can reason about from first principles, and that dataset cannot be forked away.

---

## 1. Edge-case register

### A. Hardware detection

| # | Edge case | Handling |
|---|---|---|
| A1 | RAM: total vs available vs post-settle | Linux `MemAvailable`; Windows `ullAvailPhys`; macOS `host_statistics64` (free+inactive+purgeable). Never `MemFree`. |
| A2 | iGPU steals 1–2 GB shared RAM | Detect carve-out (BIOS-reserved vs dynamic); subtract from budget |
| A3 | Apple Silicon unified memory | RAM *is* VRAM — different math; respect `iogpu.wired_limit_mb` (~75% default) |
| A4 | Windows reports "shared system memory" as VRAM | Only count **dedicated** VRAM; shared is a lie |
| A5 | Optimus / MUX / eGPU — dGPU may be powered off | Detect actual active adapter; report both states |
| A6 | P-core / E-core / big.LITTLE; hyperthreading | Report physical P-cores. Using all logical cores is usually **slower** — recommend, don't max |
| A7 | **WSL2 / VM / container** | Detect (`/proc/version`, DMI, cgroup limits). WSL2 defaults to 50% of host RAM. Warn loudly + link `.wslconfig` fix |
| A8 | Models live on a different volume than the OS | **Benchmark the model directory's volume**, not "the disk". Ask if unset. |
| A9 | Model dir is on USB / network / OneDrive / iCloud / Dropbox | Detect and hard-warn. Sync folders are catastrophic and silent. |
| A10 | Page cache fakes 10 GB/s reads | O_DIRECT / F_NOCACHE / FILE_FLAG_NO_BUFFERING mandatory |
| A11 | **Windows Defender real-time scan** | Measurable throughput collapse. Detect exclusion status, offer the exact exclusion command |
| A12 | BitLocker / FileVault / LUKS | Detect; apply measured (not assumed) penalty — it's already in the benchmark |
| A13 | Benchmarking while throttled or on battery-saver | Detect power source + thermal state; label the report `[measured on battery]` and offer re-run |
| A14 | No AVX2 (pre-2013) / ARM SBC / 32-bit | Feature-flag detection; explicit "your CPU predates the instruction sets these runtimes need" |
| A15 | Single vs dual channel RAM | **Never assume** — the STREAM benchmark catches it. 2× error if you guess. |

### B. Prediction model

| # | Edge case | Handling |
|---|---|---|
| B1 | IQ-quants dequant slower per byte than K-quants | Per-quant-family `η` coefficient, calibrated |
| B2 | MoE vs dense active-parameter math | Separate formula path; requires per-model `n_experts`, `n_active`, `shared_expert_size` |
| B3 | GQA vs MHA changes KV by 4–8× | Per-model `n_kv_heads` from `config.json` — never infer |
| B4 | tok/s degrades as context grows | Always state the context the prediction assumes (default 2K); provide a decay curve |
| B5 | Runtime defaults differ (Ollama ctx=4096, LM Studio, llama.cpp) | Predict **per runtime**, state assumed settings explicitly |
| B6 | Flash attention / KV quant availability changes context math ~4× | Detect runtime version + capability; show both with and without |
| B7 | Partial GPU offload (`n_gpu_layers`) | Model the blend; **recommend the optimal layer split** — high-value output nobody else gives |
| B8 | Model absent from database | Fetch `config.json` from the HF repo id (~2 KB) and compute live |
| B9 | Point estimates create false confidence | **Always emit ranges** (`~9–13 tok/s`) with a confidence tier based on calibration density for that hardware class |
| B10 | Prediction is simply wrong | Ship `zc verify` — runs the real model for 30 s, prints predicted vs actual, offers to submit. Turns errors into training data. |

### C. Product, trust, distribution

| # | Edge case | Handling |
|---|---|---|
| C1 | Hardware probing feels invasive | **Zero network by default.** Sharing requires an explicit `--share`. Report contains no hostname, username, serial, MAC, or IP. Print exactly what's collected before sending. |
| C2 | Windows SmartScreen blocks unsigned binaries | Ship via `winget` + PowerShell installer early; buy an EV cert (~$400/yr) before any real push |
| C3 | `npx` needs Node, which many targets lack | Primary: single static binary + `curl \| sh` + `winget` + `brew`. npx is a convenience path, not the path. |
| C4 | Corporate machines: no admin, proxies, air-gapped | No-install portable binary; honor `HTTP(S)_PROXY`; full offline mode with a bundled DB |
| C5 | Browser version can't read real hardware | Web = result-card rendering + download CTA. Never fake a browser-side "scan" — it destroys trust. |
| C6 | Fake/garbage submissions poison the DB | Plausibility checks (bandwidth vs core count vs arch), outlier quarantine, human review on PRs |
| C7 | DB staleness — models ship weekly | Nightly GitHub Action ingesting HF `config.json` + GGUF metadata; auto-PR |
| C8 | Ollama ships this natively | Defensibility is the **dataset + community DB**, not the code. Also: be a good citizen and offer it upstream. |
| C9 | High-end users feel excluded | Works for everyone; the *positioning* is low-end, the tool isn't |
| C10 | Ollama already installed with models | Detect it — offer instant real calibration from models already on disk. Best possible first-run experience. |

---

## 2. Repo & architecture

```
zerocloud/
  crates/
    zc-probe        hardware detection (per-OS cfg backends)
    zc-bench        RAM / disk / compute / thermal micro-benchmarks
    zc-model        prediction math, KV & budget calculators
    zc-db           model catalog loader, offline bundle, HF config fetch
    zc-report       terminal render, JSON, share-card encoding
    zc-cli          `check | verify | share | doctor`
  data/
    models/*.json         community-editable catalog (PR surface)
    calibration/*.jsonl   measured submissions
    profiles/*.json       known hardware quirks (iGPU carve-outs etc.)
  web/                    static result cards (GitHub Pages / Vercel static)
  .github/workflows/      nightly model ingest, submission validation, release
```

**Zero backend.** Catalog is JSON in the repo behind a CDN. Submissions are GitHub Actions opening PRs. Result cards are hash-encoded in the URL. Ops cost: $0. Scales infinitely. And `data/` being the contribution surface is what converts users into contributors.

**Language:** Rust. Single static binary, no runtime deps, trivially cross-compiled, and `zc-bench` needs O_DIRECT/F_NOCACHE anyway.

---

## 3. Program phases

### Phase 0 — Validate the physics (Week 1) ⛔ GATE
Build `zc-bench` + the prediction math. Validate against **≥5 real machines** spanning 8 GB Windows, Apple Silicon, an old Intel Mac, a Linux desktop, and something DRAM-less.

**Gate: median prediction error < 25% on decode tok/s.** If you can't hit that, the product doesn't work and you find out in a week rather than a quarter.

### Phase 1 — Core CLI (Weeks 2–4)
Full probe with edge cases A1–A15. Catalog of 40 models × relevant quants. `zc check` with ranges + confidence tiers + KV/context math + GPU-layer recommendation. `zc verify` for calibration. Binaries for macOS/Windows/Linux.

### Phase 2 — Share loop + web (Weeks 5–6)
Static result cards, share URLs, submission via GitHub Action, README with a terminal GIF above the fold, offline bundle for air-gapped use.

### Phase 3 — Launch (Weeks 7–8)
r/LocalLLaMA → HN → one YouTube reviewer on an old laptop. Then a month of answering "will this run on my X" threads by hand. Unglamorous; it's what works.

**Target: 500–2,000 stars, 2,000 runs, 300 calibration submissions.**

### Phase 4 — The guard (Months 3–5)
Once people trust you with hardware access, add what they *actually* need next: thermal + SSD governor, SSD Health Ledger from SMART, watchdog, OOM prevention. Positioning: *the local AI that cannot break your laptop.* Wraps Ollama/llama.cpp — still not your engine.

### Phase 5 — The engine (Months 6–15)
Now every `✗ Qwen3-30B-A3B — won't fit` in every report you've ever printed becomes `✓ with ZeroCloud streaming, ~9 tok/s`. You have distribution, a calibration dataset that tells you exactly which hardware benefits, and trust. This is when the work from `zerocloud_spec_v2.md` gets built — with users waiting for it.

---

## 4. Success metrics

| Milestone | Metric | Target |
|---|---|---|
| Phase 0 gate | median prediction error | < 25% |
| Phase 1 | error after 300 calibrations | < 15% |
| Phase 3 | GitHub stars / runs | 1,000 / 2,000 |
| Month 6 | calibration submissions | 5,000 |
| Month 6 | contributors (mostly `data/` PRs) | 25+ |
| Phase 5 entry | users whose top pick is currently a `✗` | > 40% |

That last metric is the real one — it's the measured size of the market for the engine, and it's the number that tells you whether to build it at all.

---

## 5. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Predictions wrong → instant credibility loss | **Critical** | Ranges not points; confidence tiers; `zc verify`; Phase 0 gate |
| Hardware probe reads as spyware | **Critical** | Zero network default, explicit `--share`, print-before-send, open source |
| SmartScreen kills Windows conversion | High | winget + PowerShell path early, EV cert before the push |
| Model DB rots | High | Nightly automated ingest from day one |
| Ollama ships it natively | Medium | Dataset is the moat; offer upstream; move fast to Phase 4 |
| Solo maintainer burnout | Medium | `data/` PRs carry the recurring load, not you |
