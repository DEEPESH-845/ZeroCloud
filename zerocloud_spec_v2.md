# ZeroCloud Core — Spec v2: Adversarial Review & Production Roadmap

> This document challenges, corrects, and expands `zerocloud_technical_spec.md`.
> It is written to be brutal where the original is optimistic, because the failure
> mode for this project is not "we shipped late" — it is "we shipped 0.2 tok/s,
> Hacker News called it a toy, and the mission died."

---

## 0. The Verdict Up Front

The safety engineering in v1 is genuinely good instinct. The **core performance thesis is wrong**, and it's wrong in a way that is fatal to the product if not corrected now.

**v1 claim:** 70B dense models on 8GB RAM via NVMe streaming.
**Reality:** ~0.1–0.3 tok/s. A single paragraph takes 10–25 minutes. Nobody will use it twice.

**v2 claim (the reframe):** ZeroCloud runs **30B–120B sparse MoE models at 5–15 tok/s on 8GB machines**, and 70B dense models in an explicit, honestly-labeled **batch mode**.

That reframe is not a retreat. In 2026 it's a *better* product than the original pitch, because the frontier open-weight models are MoE (gpt-oss-120b, Qwen3-235B-A22B, DeepSeek-V3-class, Llama-4-class), and MoE is the architecture that SSD streaming was born to serve. Dense 70B is the *worst possible* target for this engine. Sparse MoE is the *best possible* target. v1 aimed at the wrong model class.

---

## 1. The Physics: Where v1 Breaks

### 1.1 The bandwidth wall

Autoregressive decode is memory-bandwidth-bound. Every token requires reading **every active parameter**. For a dense model, "active" means "all of them."

| Model | Quant | Bytes/token read | Gen4 NVMe @6 GB/s | Gen5 @12 GB/s |
|---|---|---|---|---|
| Llama-70B dense | Q4_K_M | 42.5 GB | **0.14 tok/s** | 0.28 tok/s |
| Llama-70B dense | IQ2_XXS | 19 GB | 0.31 tok/s | 0.63 tok/s |
| gpt-oss-120b MoE | MXFP4 | ~2.7 GB active | **2.2 tok/s** | 4.4 tok/s |
| gpt-oss-120b MoE | MXFP4 + expert cache | ~0.6–1.0 GB | **6–10 tok/s** | 12–20 tok/s |
| Qwen3-30B-A3B MoE | Q4_K_M | ~1.7 GB active | 3.5 tok/s | 7 tok/s |
| Qwen3-30B-A3B MoE | + cache + spec-decode | ~0.3–0.5 GB | **12–25 tok/s** | 25–40 tok/s |

Add speculative decoding's *I/O amortization* (§3.5) and multiply the MoE rows by another 2–4×.

**This table is the entire product strategy.** It says: pin the dense parts, cache the hot experts, stream only cold experts, and amortize every read across multiple tokens. Everything else in this document is a consequence.

### 1.2 Why MoE changes everything

A MoE forward pass touches two very different classes of weight:

- **Hot core (dense-every-token):** embeddings, attention (Q/K/V/O), layernorms, routers, shared/always-on experts. For a 120B-class MoE this is typically **1.5–2.5 GB quantized**. This *fits in RAM on an 8GB machine.* Pin it. Never read it from disk again after warmup.
- **Routed experts:** the other 55–60 GB. Only top-k (usually 4–8 of 64–128) fire per layer per token.

Expert activation is **heavily skewed** — a minority of experts serve a majority of tokens, and the skew is strongly conditioned on domain (a coding conversation hits a stable expert subset). An LRU-with-frequency-decay cache of 3–4 GB routinely achieves 50–80% hit rates within a single conversation.

So the real steady-state SSD traffic for a 120B model is not 60 GB/token. It's **300 MB – 1 GB/token.** That is a shippable product on a $400 laptop.

### 1.3 The corollary nobody states

**Prefill is nearly free; decode is expensive.** Processing a 4,000-token prompt reads the weights *once* (high arithmetic intensity, batched over all positions). Generating 200 tokens reads them 200 times.

This has a profound product implication: **ZeroCloud is architecturally optimized for read-heavy, write-light workloads** — RAG, summarization, extraction, document Q&A, classification, translation. It is architecturally *bad* at long creative generation.

Which happens to be exactly the shape of the education and clinical use cases in Pillar 2. Lean into it. Make "answer from my documents in 3 sentences with citations" the hero demo, not "write me a story."

---

## 2. Architectural Corrections to v1

### 2.1 ✗ `mmap` + `MADV_DONTNEED` is the wrong I/O primitive

Three separate problems:

1. **`mmap` page faults cannot saturate NVMe.** Faults are synchronous, per-4KB, and serialize on the faulting thread. NVMe needs QD32–64 to reach rated throughput. Real-world mmap fault throughput on a 6 GB/s drive is often **0.8–1.5 GB/s** — you leave 75% of the drive on the table, which directly multiplies into your tok/s.

2. **`MADV_DONTNEED` does not do what the spec says.** On a `MAP_PRIVATE` file mapping under Linux, it zaps your PTEs but leaves the *clean file pages sitting in the kernel page cache*. You have not freed memory; you've just made yourself fault again. Truly evicting requires `posix_fadvise(POSIX_FADV_DONTNEED)` on the file (coarse, racy, and a **no-op on macOS**). On macOS `MADV_DONTNEED` semantics differ again and interact unpredictably with memory compression.

3. **You're fighting the kernel for a job you should own.** The page cache is a general-purpose heuristic. You know the exact access pattern.

**✓ Correction — own the memory, bypass the cache:**

```
Linux:   io_uring + O_DIRECT, registered buffers, QD64, 128KB–1MB reads
Windows: FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED (+ DirectStorage on Win11)
macOS:   fcntl(F_NOCACHE=1) + fcntl(F_RDAHEAD=0), pread() on a 8–16 thread pool
         (no io_uring exists; a thread pool is the only way to get queue depth)
```

Read into a **user-space slab arena** you allocated once. You control residency, eviction, and alignment. The OS page cache never doubles your footprint. This also makes the "keep RAM under 4GB" guarantee *actually enforceable* instead of advisory.

Note: this makes "Zero-Copy Paging Engine" a slight misnomer. With O_DIRECT into registered buffers it's near-zero-copy (DMA straight into your arena). Call it the **Weight Streaming Engine** and be accurate.

### 2.2 ✗ The TBW analysis is half-right and misses the real write sources

Read-only mapping is correct — clean file pages are discarded, never swapped. Good. But:

- **`mlock()` will fail for most users.** Default `RLIMIT_MEMLOCK` on Linux is frequently 8 MB (sometimes 64 KB). You need `CAP_IPC_LOCK`, a raised ulimit, or — far better — **just don't rely on mlock**. Size the KV cache to provably fit in the resident budget and it will never be swapped in practice. Use mlock opportunistically, degrade silently, never hard-fail on it.
- **The real TBW eater is the model pipeline, not inference.** A 60 GB download + quantization + repack can write 150 GB in one afternoon. A user who tries five models has written 750 GB. Fix: **content-addressed model store** (BLAKE3), dedup shared tensors, resumable downloads, stream-transform during download (never write an intermediate file), and never re-convert.
- **Windows pagefile** is a separate beast — must be handled explicitly.
- **Reads are not perfectly free.** Sustained heavy reads on the same NAND blocks trigger **read-disturb mitigation**, where the controller internally rewrites blocks. It's small, but "we write zero bytes" is an overclaim. Don't make claims you can't measure.

**✓ Add: the SSD Health Ledger.** Read SMART / NVMe `Data Units Written` (via `nvme-cli`, `smartctl`, IOKit, or Windows storage APIs) at session start and end. Show the user, in the UI:

> *This session ZeroCloud read 84 GB and your drive recorded 0.4 GB written.
> Lifetime ZeroCloud writes: 12 GB. Your drive is rated 600 TBW and is at 3.1%.*

No local inference tool ships this. It converts the #1 objection ("won't this kill my SSD?") into the #1 trust signal, with *measured* numbers instead of promises.

### 2.3 ✗ The thermal model targets the wrong component and the wrong sensors

**Wrong component:** This engine is **I/O-bound, not compute-bound.** The CPU spends most of its time blocked on NVMe. The component that actually overheats under sustained 6 GB/s reads is **the SSD itself** — Gen4 controllers draw 5–8 W, hit 70–85 °C in a thin laptop chassis, and thermal-throttle to a third of rated speed. That throttle is invisible to CPU temperature monitoring and directly craters your tok/s. **v1 does not mention SSD thermals at all. This is the real thermal risk in this design.**

**Wrong sensors:** SMC temperature keys on Apple Silicon are undocumented, version-unstable, and `powermetrics` needs sudo. `MSAcpi_ThermalZoneTemperature` on Windows is unimplemented on most consumer laptops. Many Linux laptops expose no usable thermal zone.

**✓ Correction — measure the *effect*, not the cause:**

| Signal | Availability | Use |
|---|---|---|
| `ProcessInfo.thermalState` (macOS) | No privileges, always works | Primary macOS signal |
| NVMe SMART composite temp + throttle counters | Cross-platform, exact | **Primary SSD signal** |
| Achieved GB/s vs. calibrated baseline | Always | Throttle detection by inference |
| Achieved tok/s vs. rolling baseline | Always | Ground-truth "am I being throttled" |
| RAPL / powermetrics / hwmon | Sometimes | Bonus refinement |
| Fan RPM delta | Sometimes | UX signal ("your fans just spun up") |

**✗ Micro-sleeps between matmuls are the wrong actuator.** 5 ms sleeps mid-matmul destroy cache locality and L2 residency, costing more than they save. Correct actuators, in order of effectiveness:

1. **Thread count reduction** — power scales superlinearly with core count and frequency. Going 8→4 threads often costs 25% speed for 55% power. This is the single best knob.
2. **Core affinity** — pin to E-cores/efficiency cores in Eco mode (Apple Silicon, Intel P/E, ARM big.LITTLE).
3. **I/O rate limiting** — a token bucket on the read queue. This is how you cool the *SSD*, and it's a knob nothing else in this space has.
4. **Duty cycling at layer boundaries** — sleep between layers, never inside a kernel.

Power profiles become: `Silent` / `Balanced` / `Unleashed` / `Plugged-in-only`, each a tuple of (threads, affinity, IO budget GB/s, duty ratio, max draft length).

### 2.4 ✗ "Refuse to run if disk < 1000 MB/s" is user-hostile

Never refuse. **Predict, disclose, and let the user choose.**

The first-run calibration (§4.2) measures the user's actual sequential and random read at QD1 and QD32, then shows a *predicted tok/s per model* **before** downloading 40 GB. A user on a 450 MB/s SATA SSD in Lagos should see "Qwen3-30B-A3B: ~2 tok/s — usable for batch jobs, slow for chat" and get to decide. That's respect. Refusal is paternalism, and it excludes exactly the users this project exists for.

Also: DRAM-less SSDs (ubiquitous in cheap laptops — the target market) have catastrophically worse random read. The calibrator must detect this specifically and bias toward larger, more sequential access patterns for those users.

### 2.5 ✗ The Compute Dispatcher's premise is backwards on integrated GPUs

v1: *"If a layer is small enough, route to GPU."*

On an 8 GB machine, the GPU is almost always an iGPU sharing the *same* system RAM. Routing to the GPU doesn't add memory — **it subtracts it**. And because we're I/O-bound, faster matmul barely moves the needle.

**✓ Correction:**
- **Apple Silicon:** unified memory, so Metal is nearly always right — no copy cost, big bandwidth win. Use it.
- **Discrete GPU (even 4 GB):** hugely valuable, but not for "small layers" — use it as a **VRAM-resident cache tier for the hot core + hottest experts**. A 4 GB GTX 1650 holding the dense core is transformative.
- **Intel/AMD iGPU:** usually not worth the RAM. Default off, benchmark-gated on.

Reframe the dispatcher as a **Tier Placement Planner** over a memory hierarchy: `VRAM → RAM (pinned) → RAM (expert cache) → NVMe → (network)`. Its job is deciding *where each tensor lives*, not which ALU runs it.

### 2.6 ✗ KV cache will blow the budget long before the weights do

v1 mentions ring buffers in Phase 4. It's a Phase 1 problem.

Llama-3-70B KV at fp16: 80 layers × 2 × 8 KV heads × 128 dim × 2 bytes = **320 KB/token**.
- 8K context → 2.6 GB
- 32K context → **10.5 GB** — larger than the machine.

**✓ Required from day one:**
- **KV quantization to Q8 (2×) and Q4 (4×)** — 32K context drops to ~2.7 GB. Non-optional.
- **Paged/block KV allocation** (vLLM-style, 16-token blocks) so fragmentation doesn't waste 30% and so multiple agent sessions can share the pool.
- **Prefix caching** — RAG systems resend the same system prompt and same retrieved chunks constantly. Cache the KV for shared prefixes across turns and across agents. On this architecture prefix caching saves *disk reads*, not just FLOPs. Enormous win.
- **Attention sinks + sliding window** for graceful overflow, not naive middle-eviction (v1's "evict middle tokens" quietly destroys RAG answers whose evidence lives in the middle).
- **KV offload to SSD only in explicit long-context batch mode**, into a *preallocated, rewritten-in-place* scratch file, with the write volume charged against a visible TBW budget. This is the one place ZeroCloud writes meaningfully, and the user must consent.

### 2.7 ✓ Add: streaming-optimized model format (`.zcm`)

GGUF's tensor order is not execution order, and its alignment is not I/O-friendly. Reading a 60 GB MoE through GGUF's layout produces scattered mid-size reads.

**`.zcm` = GGUF superset with:**
- Tensors laid out in **exact forward-pass execution order**; experts grouped **expert-major within layer** so a top-k selection is k contiguous reads.
- **4 KB alignment minimum, 128 KB target read granularity** — matches NVMe optimal transfer size.
- A **hot-core section** at the head of the file, contiguous, so warmup is one big sequential read.
- **Per-expert offset index + per-expert BLAKE3** for integrity and cache validation.
- **Mixed-precision by criticality:** attention/router/shared-experts at Q6–Q8 (they're pinned in RAM anyway, so precision is free), routed experts at Q3–Q4 (they're the streamed bulk, so bits are literally time). This is a quality-per-second win no RAM-resident engine has any reason to invent.
- Optional **per-block compression** (LZ4/zstd-1) where decompress speed > disk speed — on a 500 MB/s SATA drive this is a straight 1.5–2× throughput multiplier. On Gen5 it's a loss. Calibration decides.

Ship a `zc convert` that produces `.zcm` from any GGUF/safetensors, and **upstream the layout proposal to the GGUF community** rather than fragmenting the ecosystem.

### 2.8 ✓ Add: security — currently absent from v1, and disqualifying for the medical pillar

Non-technical users will download arbitrary model files from the internet. That is a binary-parser attack surface pointed at your users.

- **Parser hardening:** `#![forbid(unsafe_code)]` in all format parsing, `cargo-fuzz` in CI, OSS-Fuzz enrollment. (llama.cpp's GGUF reader has shipped heap-overflow CVEs. Assume you will too unless you fuzz.)
- **Chat templates are Jinja — that's an RCE vector shipped inside model files.** Use `minijinja` with globals/filters stripped to an allowlist, executed under a time and memory bound.
- **Signed model manifests.** A community-run signing registry with pinned BLAKE3 digests; UI warns loudly on unsigned models.
- **Tool/agent sandbox** (§5.2) — capability-scoped, filesystem-jailed, no network by default.
- **Provable no-network mode.** Ship a build where the inference process runs under seccomp-bpf (Linux) / App Sandbox with no network entitlement (macOS) / AppContainer (Windows) such that outbound sockets are *impossible at the kernel level*, not merely unused. Model downloads happen in a separate, short-lived, clearly-labeled helper process.

That last item is the entire medical/legal go-to-market in one feature. "Trust us, we don't phone home" is worthless. "The kernel will not let this process open a socket, here's the audit script" is a procurement-grade claim.

### 2.9 ⚠ Tech stack: right language, wrong place to spend the effort

Rust for the engine is correct — you're doing manual memory paging with an ownership discipline and lifetime guarantees are worth real money here.

But **Candle is not competitive with GGML on CPU quantized matmul or on Metal**, and writing your own AVX-512/NEON quantized kernels is a 12-month detour that produces something worse than what already exists.

**The differentiator is the I/O and scheduling layer, not the matmul.** So:

- **Build ZeroCloud as a streaming weight provider + scheduler that plugs into `ggml` as its compute backend** (via `ggml-sys` FFI). You inherit Metal, CUDA, Vulkan, AVX-512, and NEON kernels for free, and you ship in months instead of years.
- Upstream a **`ggml` "weight provider" hook** — an interface where a tensor's data is fetched on demand rather than assumed resident. This benefits the whole ecosystem and makes ZeroCloud a good citizen rather than a fork.
- Keep a Candle-based pure-Rust path as a build feature for platforms GGML doesn't reach and for people who need a no-C-dependency build. Don't make it the default.

Rust owns: paging engine, expert cache, scheduler, governor, server, RAG, agent runtime, CLI/GUI backend. C++ owns: the ~15 hot kernels. That's the right seam.

---

## 3. Pillar 1 — Production Readiness & Usability

### 3.1 Distribution: one click, actually

| Platform | Artifact | Notes |
|---|---|---|
| macOS | Notarized universal2 `.dmg` + Homebrew cask | Notarization is mandatory or Gatekeeper blocks it |
| Windows | Authenticode-signed MSIX + `winget` | **Budget an EV cert (~$400/yr)** — unsigned = SmartScreen wall = dead |
| Linux | AppImage (primary), Flatpak, `.deb`/`.rpm`, `curl \| sh` | AppImage first: works on old distros, which is the target market |
| Portable | **`zerocloud-portable/` on a USB stick** | No install, no admin rights. See §4.3 |

Single static binary where possible. No Python. No CUDA toolkit prerequisite. No "first install Visual C++ Redistributable." Every prerequisite you can't eliminate is a support ticket from someone who will never file it — they'll just quit.

### 3.2 First-run: the Hardware Passport

A 20-second calibration on first launch, no questions asked:

```
✓ Storage       Samsung 980  ·  seq 3.4 GB/s  ·  QD32 rand-128K 2.9 GB/s  ·  DRAM cache: yes
✓ Memory        8 GB total  ·  5.1 GB available  ·  budget set to 4.6 GB
✓ Compute       8 cores (4P/4E)  ·  AVX2  ·  Iris Xe iGPU (shares RAM — disabled)
✓ Thermals      SSD idle 41°C  ·  sustained-read headroom: good
✓ Power         On battery  ·  Balanced profile selected

Recommended for your machine:
  ★ Qwen3-30B-A3B  (Q4)   ~14 tok/s   17 GB download   ← best experience
    gpt-oss-120b   (MXFP4) ~6 tok/s   61 GB download   ← most capable
    Llama-70B      (IQ2)   ~0.4 tok/s 19 GB download   ← batch mode only
```

Predicted tok/s **before** the download, derived from measured hardware — not from a marketing page. Cache the passport; re-calibrate on hardware change.

**Then make it a community asset:** opt-in, anonymized submission builds a **public hardware compatibility database** (`hardware.zerocloud.dev`) — "what does ZeroCloud do on a Lenovo E14 with 8 GB?" That database is a genuine moat, a marketing engine, and a gift to the community, all from one checkbox.

### 3.3 Zero-configuration means the config file is a *fallback*, not a step

- Auto-select model, quant, thread count, KV size, power profile, and expert-cache size from the passport.
- **Continuous auto-tuning:** if achieved tok/s drops >25% below baseline, diagnose (SSD thermal throttle? another app eating RAM? battery saver kicked in?) and adapt automatically, then *tell the user in one plain sentence what happened.*
- `config.toml` exists and is documented, but a user should be able to go from download to first token without ever seeing it.

### 3.4 Bulletproof errors: a versioned error catalogue

Every failure gets a stable code, a plain-language cause, and a **one-click fix action**:

```
ZC-1042 · Not enough free memory to start

  Chrome is currently using 3.8 GB. ZeroCloud needs 4.6 GB and
  only 2.9 GB is available.

  [ Use the smaller model instead ]   [ Retry after closing Chrome ]   [ Details ]
```

Not `Error: mmap failed: ENOMEM`. Never a stack trace in the primary UI.

Supporting machinery:
- **Watchdog supervisor process.** The engine runs as a child. If it hangs, OOMs, or segfaults, the watchdog kills it, restarts in a safe profile (smaller model, fewer threads), and **restores the conversation**. The user sees "recovered — switched to Balanced mode," not a crash.
- **Pre-flight RAM reservation.** Allocate and touch the arena at startup. Fail fast and gracefully at t=0 instead of freezing the machine at t=90s.
- **Crash-only design.** All state (conversations, indexes, settings) in SQLite with WAL. Kill -9 at any moment must be safe.
- **`zc doctor`** — one command that produces a complete, shareable diagnostic bundle with secrets and document content redacted. This turns unreproducible bug reports into fixable ones.
- **Deterministic mode** (fixed seed, fixed thread count, disabled fast-math) for reproducible outputs. Required for regulated/clinical evaluation.

### 3.5 Model delivery for bad networks — this is a first-class engineering problem

A 40 GB download assumes infrastructure the target user doesn't have.

- **Resumable, chunked, BLAKE3-verified** downloads. Survive a 12-hour disconnect.
- **Delta updates.** Model v1.1 should ship as a 400 MB diff, not a 40 GB re-download.
- **LAN peer discovery (mDNS).** In a school lab, machine #2 pulls from machine #1 at gigabit instead of from the internet at 2 Mbps. This is ~200 lines of code and it changes deployability completely.
- **BitTorrent/IPFS fallback** for mirrors.
- **Sneakernet bundles.** `zc export --to /Volumes/USB` produces a self-contained, signed, offline-installable bundle: engine + model + knowledge packs. `zc import` on the target machine. **For a large fraction of the intended beneficiaries, USB is the network.**
- **Bandwidth-aware scheduling** — download overnight, pause on metered connections (detect via OS APIs), hard cap on data usage.

### 3.6 Release engineering

Staged rollout (1% → 10% → 100%), instant rollback, signed update manifests, and a **performance regression gate in CI**: a nightly benchmark on fixed reference hardware that blocks any merge regressing tok/s by >5%. On a project whose whole value is throughput, perf regressions are correctness bugs.

---

## 4. Pillar 2 — Societal Value

The mission statement in v1 is right but abstract. Here is what makes it real, and each item below is a shippable artifact rather than a sentiment.

### 4.1 Knowledge Packs — the offline education primitive

Define a signed bundle format: **curated corpus + prebuilt vector index + curriculum-tuned system prompt + optional LoRA adapter + evaluation set.**

```
knowledge-packs/
  ss2-physics-ng.zcpack     Nigerian SSCE physics · 2.1 GB · Hausa/Yoruba/English
  wikipedia-simple.zcpack   Simple English Wikipedia · 4.8 GB
  openstax-bio.zcpack       OpenStax Biology 2e · 900 MB
  who-primary-care.zcpack   WHO primary-care guidelines · 1.4 GB
  farmer-agronomy-ke.zcpack Kenyan smallholder agronomy · 600 MB
```

Why this is the right unit: the pack — not the model — is what makes the AI *locally useful*. A pack is small enough to ship on a phone's SD card. Packs are community-authorable. A teacher in Kerala can build one for their syllabus. **The index ships prebuilt**, so a slow laptop doesn't spend six hours embedding 5 GB of text.

Crucially, this exploits §1.3: RAG over a local pack is a *prefill-heavy, decode-light* workload, which is exactly where this engine is strong. The mission and the physics point the same direction.

### 4.2 Language equity via an adapter ecosystem

Frontier open models are mediocre-to-terrible in most of the world's languages. Full fine-tunes are undownloadable for the people who need them; **LoRA adapters are 50–300 MB** and download fine on 3G.

- Adapters stay **pinned in RAM** (they're tiny) and merge at compute time — the streaming architecture handles them for free.
- Ship `zc finetune` — overnight local QLoRA on a small model, on the user's own machine, on their own data. A community can improve its own model without a GPU, without cloud, and without giving anyone their corpus.
- Run an open adapter registry with provenance and eval scores.

This is the single highest-leverage equity feature in the document. It moves the ceiling for languages the market will never serve.

### 4.3 Clinic-in-a-box / privacy-critical deployments

- **Attested no-network mode** (§2.8) — kernel-enforced, with a published audit script. This is what makes it procurable by a clinic, a legal aid office, or a domestic violence shelter.
- **Grounded-or-silent mode:** a decoding constraint where every factual claim must cite a retrieved local document, and the model is required to abstain when retrieval is empty. Enforced structurally via constrained decoding (§5.3), not via prompt-begging.
- **Encrypted-at-rest** conversation and index store (OS keychain-backed).
- **Audit log** of every query and every document retrieved — for institutions that need it.
- **Positioning discipline:** documentation assistant, guideline lookup, patient-language translation, note summarization. **Not diagnosis.** Under FDA and EU MDR, "provides a specific diagnosis or treatment recommendation" is a regulated device. Ship an explicit intended-use statement, refuse-to-diagnose behavior in the default clinical persona, and get a lawyer before the medical landing page goes live. Getting this wrong doesn't just risk liability — it discredits the whole project.

### 4.4 Developer empowerment in emerging markets

- **OpenAI-compatible endpoint** means every existing tool, tutorial, and SDK works unmodified on day one. Non-negotiable.
- **LAN server mode:** one 32 GB machine in a lab serves 20 thin clients over mDNS-discovered HTTP. Continuous batching (§5.2) makes this genuinely efficient — 20 concurrent requests amortize the same weight reads. **This is the highest-value deployment shape for a university in a low-income country, and it falls out of the architecture almost for free.**
- **Cost narrative with real numbers:** a developer doing 2M tokens/day pays ~$0 locally vs. real money on any cloud, and doesn't need a credit card or a payment rail that works in their country. For a lot of people this isn't cost optimization — it's the difference between access and no access.

### 4.5 Sustainability and e-waste

Every 2018 laptop that stays useful is a laptop not landfilled. Quantify it: publish measured Wh/token for ZeroCloud on old hardware vs. the amortized datacenter figure. This is a real, defensible ESG story and it unlocks a funding category most infra projects can't touch.

### 4.6 Funding — because this cannot be a hobby

The engineering in this document is 2–4 person-years. Name the targets now: **NLnet / NGI Zero** (fits perfectly, funds exactly this), **Sovereign Tech Fund**, **Mozilla Technology Fund**, **UNICEF Innovation Fund**, **Gates Foundation** (education/health packs), **Sloan Foundation** (open research software). Governance: permissive core (Apache-2.0) so it can be embedded everywhere, with a foundation or fiscal host from early on so institutions can actually give you money.

---

## 5. Pillar 3 — 2026-Era Engine Features

### 5.1 Native Local-RAG (in the engine, not bolted on)

Because prefill is cheap and decode is expensive, RAG is *architecturally favored* here. Build it in:

- **Embedded vector store** (SQLite + `sqlite-vec`, or usearch/HNSW) — no separate server, no Docker.
- **Hybrid retrieval:** BM25 + dense, reciprocal-rank fused. Beats pure-dense on the sparse, jargon-heavy corpora (medical, legal, curriculum) that matter most here.
- **Late-interaction reranking** (ColBERT-style, tiny model, pinned in RAM) — large quality gain for ~200 MB.
- **Embedding + reranker models permanently pinned.** ~600 MB total, never streamed.
- **Ingestion pipeline:** PDF (with OCR for scans — non-negotiable for the Global South, where documents are photographs), DOCX, EPUB, HTML, Markdown, code. Semantic chunking, not fixed-width.
- **Filesystem watcher** with incremental reindexing. Point it at a folder, forget about it.
- **Prefix-cached retrieval contexts** (§2.6) — the same retrieved chunk across turns costs zero re-read.
- **Citations by construction**, with clickable spans back into the source document.

### 5.2 Local agentic orchestration

- **Continuous batching from day one.** Multiple agents running in parallel share one pass over the weights. On a streaming engine this isn't a throughput nicety — **it's a 3–5× multiplier on the whole agent use case**, because weight I/O is amortized across concurrent requests. Nothing else in local inference has this incentive structure.
- **Capability-scoped tool sandbox.** Filesystem jail, no network by default, per-tool explicit grants, resource limits, full audit trail. An agent that can run shell commands on a non-technical user's laptop is a loaded weapon; treat it that way.
- **MCP client *and* server.** Client so ZeroCloud can use the 2026 tool ecosystem; server so ZeroCloud is a tool other agents can call.
- **Durable, resumable agent sessions** in SQLite. On slow hardware, a multi-step agent run takes real time; it must survive a lid close.
- **Cascade routing (0.5B → 8B → 120B).** A confidence/complexity classifier keeps the majority of turns on a model that's fully RAM-resident and fast, escalating only when needed. This is what makes the product *feel* fast in daily use, and it's a bigger perceived-performance win than any kernel optimization.

### 5.3 Structured output & constrained decoding

Non-negotiable for reliability with aggressively-quantized models. GBNF/`llguidance`-style grammar constraints, JSON-Schema-to-grammar compilation, regex constraints, and enum forcing. A Q3 model with a hard grammar constraint beats an unconstrained Q5 model at every extraction task — **so constrained decoding buys you a whole quantization level, which on this architecture converts directly into tokens per second.**

### 5.4 Voice — full duplex, and the latency problem has an architectural answer

- Streaming ASR + small TTS, both **pinned in RAM** (~1 GB combined), with VAD and barge-in.
- **The hard problem is time-to-first-token**, which is this engine's structural weakness. The answer is cascade + speculation: the pinned draft model produces the opening of the response *instantly* and TTS starts speaking it while the big model streams in behind. Handle the trivial turns entirely on the draft model. Perceived latency collapses even though the big model is still slow.
- Wake word, fully offline. Voice is the accessibility story — for low-literacy users, users with disabilities, and hands-busy contexts (a clinician, a farmer, a mechanic), voice isn't a feature, it's the interface.

### 5.5 Multimodal

Vision encoders are 300 MB–1 GB and run **once per image**, not once per token. On a streaming architecture, **vision is nearly free** — pin the encoder and the marginal cost of an image is negligible.

That's a strategic gift: photograph a textbook page, a whiteboard, a prescription, a machine part, a crop leaf. Document understanding via camera is the highest-value multimodal use case in exactly the markets Pillar 2 targets, and it's the cheapest thing this architecture does.

### 5.6 Speculative decoding — the crown jewel, and v1 undersells it badly

v1 treats the draft model as "reduces the frequency of SSD fetches." Correct, but it buries the real point:

> **On an SSD-streaming engine, speculative decoding is not a compute optimization. It is an I/O amortization primitive.** Verifying k draft tokens in one forward pass reads the weights **once for k tokens.** With 5 accepted tokens per verification, you have cut SSD traffic per token by 5×. On a RAM-resident engine speculation buys maybe 2×. **Here it buys the whole product.**

That reframe justifies going much further than a draft model:

1. **EAGLE-3 / Medusa-style self-speculation heads** — a few hundred MB, pinned, higher acceptance rates than a separate draft model because they see the target model's hidden states.
2. **Prompt-lookup / n-gram decoding** — zero parameters, zero cost, and extremely effective for RAG and code, where output heavily copies input. Free multiplier on the flagship use case.
3. **Tree/multi-candidate verification** — verify a branching tree of candidates in one pass, raising accepted-tokens-per-read.
4. **I/O-aware adaptive draft length** *(novel contribution)* — measure the live bottleneck. When SSD-bound, **lengthen** the draft, because longer drafts amortize weight reads better; when compute-bound, shorten it. A closed-loop controller on draft depth driven by measured queue latency. No RAM-resident engine has any reason to invent this, which makes it defensibly ZeroCloud's.
5. **Draft-guided expert prefetch** *(the big one)* — the draft model's forward pass reveals which experts the target model will likely route to, several tokens ahead. Use it to issue prefetch reads *before* they're needed, hiding SSD latency entirely behind compute. Combined with a learned router-predictor (predicting layer N+1's routing from layer N's logits), this is what turns "streaming" into "feels resident."

Items 4 and 5 are original research contributions with real publication value, and they're the technical heart of why ZeroCloud exists.

### 5.7 Time-shifted inference — turn the weakness into a product

Big dense models are slow. Stop apologizing and productize it:

**Job queue mode.** "Summarize these 200 PDFs." "Answer these 40 exam questions with citations." Queue it, close the lid, get a notification in the morning. Scheduled to run while charging, while idle, while thermals are cool. Full progress and resumability.

For a student with a 6-year-old laptop and no cloud budget, "70B-quality answers overnight, free" is not a consolation prize — it's an offer nobody else makes.

### 5.8 Other 2026 table stakes

- **Instant model hot-swap** — nothing is truly "loaded," so switching models is a cache flush, not a 90-second reload. This is a genuine UX advantage over every RAM-resident engine and it should be demoed prominently.
- **Persistent encrypted user memory** across sessions.
- **Per-conversation, per-agent, per-pack persona/adapter binding.**

---

## 6. Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| **Bandwidth wall makes flagship models unusable** | Critical | Pivot the headline to MoE; publish honest tok/s tables; ship batch mode for dense |
| **Overpromise → "0.2 tok/s" reviews kill credibility** | Critical | Never advertise a number the calibrator can't predict on the user's own machine |
| **SSD thermal throttling craters real-world perf** | High | I/O rate limiting as a first-class governor actuator; SMART temp monitoring |
| **"You destroyed my SSD" narrative** | High | SSD Health Ledger with measured SMART deltas; conservative defaults; publish methodology |
| **OS vendors ship native local inference** | High | Compete on model *size*, cross-platform reach, privacy attestation, and packs — not on being "a local LLM runner" |
| **Malicious model file → RCE on non-technical users** | High | Fuzzed parsers, sandboxed Jinja, signed manifests, `forbid(unsafe_code)` |
| **Medical positioning → regulatory exposure** | High | Intended-use statement, refuse-to-diagnose default, legal review pre-launch |
| **Upstream model licenses (Llama community license etc.)** | Medium | Lead with Apache/MIT-licensed MoE models; per-model license surfacing in UI |
| **Maintenance burden across 3 OSes + 5 backends** | Medium | Build on ggml rather than owning kernels; CI on real reference hardware; funded maintainers |
| **40 GB download is impossible for target users** | Medium | LAN P2P, delta updates, sneakernet bundles, small-model-first defaults |

---

## 7. Architectural Roadmap v2

Timelines assume 2–3 engineers. v1's "5 weeks to speculative decoding" is off by roughly an order of magnitude; planning honestly is a safety feature.

### Phase 0 — Prove or kill the thesis (Weeks 1–3) ⛔ **GATE**

Build nothing else until this passes. This phase exists to find out if the project is possible.

- Standalone I/O benchmark harness: `io_uring`/`F_NOCACHE`/`OVERLAPPED` vs. mmap, sweeping block size and queue depth on 5+ real drives (Gen5, Gen4, Gen3, SATA, DRAM-less, USB-SSD).
- Offline expert-locality study: instrument a real MoE, capture routing traces on real conversations, measure achievable cache hit rates at 2/3/4/5 GB budgets.
- Analytical + empirical tok/s model for the whole hierarchy.

**Gate:** simulated end-to-end ≥ **5 tok/s for a 120B-class MoE on a 6 GB/s drive with a 4 GB cache.** If not met, revise the target model class before writing an engine.

### Phase 1 — Foundations: safety, storage, format (Weeks 3–10)

- Rust workspace; `zc-io` (async streaming reader with O_DIRECT/F_NOCACHE + platform backends).
- `zc-arena` (slab allocator, hard RAM ceiling, enforced not advised).
- **Hardware Safety Governor** — thermal state, SMART/NVMe temps + throttle counters, achieved-bandwidth throttle detection; actuators = thread count, affinity, **I/O rate limit**, layer-boundary duty cycling. Power profiles. ACPI/power-source events.
- **SSD Health Ledger.**
- `.zcm` format + `zc convert` from GGUF/safetensors.
- **Deliverable:** a stress harness that provably holds RAM under a hard cap and holds SSD temp under a target while saturating available bandwidth.

### Phase 2 — Streaming inference core (Weeks 8–18)

- `ggml` FFI backend; weight-provider hook (upstream the proposal).
- **Expert cache** — LFU-with-decay, per-layer budgets, VRAM tier when a discrete GPU exists.
- Hot-core pinning; tier placement planner.
- Paged KV cache with Q8/Q4 quantization, prefix caching, attention sinks.
- Correct forward pass for one MoE family + one dense family; numerical parity tests vs. llama.cpp.
- **Deliverable:** real tok/s on real hardware, published, including the bad numbers.

### Phase 3 — Making it fast (Weeks 16–28)

- Speculative decoding: draft model → EAGLE-3 heads → tree verification.
- **Prompt-lookup decoding** (cheap, big RAG/code win — do it first).
- **Draft-guided expert prefetch** + learned router predictor.
- **I/O-aware adaptive draft length** controller.
- Continuous batching; Metal / CUDA / Vulkan paths.
- Mixed-precision-by-criticality quantization in `.zcm`.
- **Deliverable:** ≥3× over Phase 2. Nightly perf-regression gate live in CI.

### Phase 4 — Product surface (Weeks 24–36)

- OpenAI-compatible server; MCP client + server; constrained decoding / JSON schema.
- Native RAG stack: hybrid retrieval, reranker, ingestion with OCR, filesystem watcher, citations.
- **Knowledge Pack format** + first three reference packs.
- Desktop app (Tauri): first-run calibration, Hardware Passport, model picker with *predicted* tok/s, one-click everything.
- Watchdog supervisor, error catalogue, `zc doctor`.
- Signed installers for all three platforms; delta updates; LAN P2P; sneakernet export.
- **Deliverable:** a non-technical user goes from download to first answer with zero configuration and zero terminal.

### Phase 5 — 1.0: agents, voice, vision (Weeks 32–48)

- Agent runtime + capability sandbox; durable sessions; cascade routing.
- Full-duplex voice with draft-model-first latency hiding.
- Vision encoder support; camera/document capture flow.
- **Attested no-network mode** with published audit script.
- Time-shifted job queue.
- LAN server mode with mDNS.
- Security: OSS-Fuzz, third-party audit, signed model registry.
- **Deliverable: ZeroCloud 1.0.**

### Phase 6 — Mission (ongoing, starts at Phase 4)

- `zc finetune` local QLoRA; adapter registry; language-equity partnerships.
- Knowledge Pack authoring tools + community program.
- Public hardware compatibility database.
- Clinical/legal deployment guides with intended-use and regulatory documentation.
- Grant applications (NLnet, STF, Mozilla, UNICEF); foundation/fiscal host.
- Papers on I/O-aware speculation and draft-guided expert prefetch.

### Workspace layout

```
zerocloud/
  crates/
    zc-io          O_DIRECT / io_uring / F_NOCACHE / OVERLAPPED streaming reader
    zc-arena       slab allocator, hard RAM ceiling, pinning
    zc-format      .zcm + GGUF/safetensors parsing (#![forbid(unsafe_code)], fuzzed)
    zc-cache       expert cache, tier placement planner, prefetcher
    zc-governor    thermal/power/IO governor, SSD health ledger, power profiles
    zc-compute     ggml FFI backend (+ optional candle backend)
    zc-engine      scheduler, KV manager, speculation, continuous batching
    zc-rag         embeddings, hybrid retrieval, reranker, ingestion, packs
    zc-agent       tool runtime, capability sandbox, MCP client/server
    zc-voice       ASR / TTS / VAD
    zc-server      OpenAI-compatible HTTP, mDNS LAN discovery
    zc-cli         zc run | convert | bench | doctor | finetune | export
  apps/desktop     Tauri app
  bench/           reference-hardware perf suite (CI regression gate)
  packs/           reference Knowledge Packs
```

---

## 8. The Three Things That Matter Most

If everything else in this document is cut, keep these:

1. **Pivot the flagship claim from dense-70B to sparse-MoE.** The physics doesn't negotiate, and MoE is where the 2026 frontier lives anyway. This turns an unusable demo into a usable product.
2. **Speculative decoding + draft-guided expert prefetch is the core research contribution.** On this architecture, speculation is I/O amortization, and that reframe is worth multiples on throughput. It's also what makes ZeroCloud defensible rather than "llama.cpp with mmap."
3. **Measured honesty as the trust product.** Predicted tok/s before download. Measured SSD writes in the UI. Kernel-attested no-network mode. Published benchmarks including the bad ones. In a category full of overclaiming, being the one project whose numbers are checkable is the whole brand — and it's the only way a clinic, a school district, or a ministry of education ever deploys you.
