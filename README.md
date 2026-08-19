# ZeroCloud

**What can this laptop actually run, and how fast?**

`zc` measures your machine in about two seconds and predicts decode speed,
time to first token, and maximum usable context for local LLMs. No account, no upload,
no network call at all.

That two seconds is measured, not estimated: 1.85s across three runs on the
Apple Silicon laptop the sample output below came from. Only the disk probe is
time-boxed, so the RAM and compute benchmarks take longer on slower hardware —
how much longer is not yet known, because nobody has measured it on a slow
machine. That is one of the things a contributed record tells us.

It is built for the machines that get told "you need a better GPU": 8 GB
Windows laptops, old Intel Macs, single-channel budget hardware, WSL2, and
Raspberry Pis. It works on a 4090 too — the positioning is low-end, the tool
isn't.

```
$ zc check --top 8

== hardware ==
  Apple M5   4P+6E   16.00 GiB total / 3.18 GiB available   unified
  /System/Volumes/Data on apfs (NVMe)
  Apple M5   integrated (shares system memory)

== measured ==
  ram          129 GB/s peak @10t   [1t:75  2t:82  4t:82  10t:129]
  compute      427 GFLOPS f32 @4t   413 GOPS int8 (0.97x)
  disk         5.02 GB/s random 128K @QD16   167K IOPS 4K
  budget       12.80 GiB on an idle machine   (1.58 GiB free right now)

== predictions ==  (Metal backend, 129 GB/s, KV at F16, 2048-token prompt)
  assumes an otherwise-idle machine
       model                        quant   decode tok/s max ctx   TTFT  conf
  OK   smollm2-360m                 Q8_0     249.5-415.8      8K   0.1s  low
  OK   qwen3-0.6b                   Q8_0     150.8-251.3     40K   0.3s  low
  OK   smollm2-1.7b                 Q8_0       53.0-88.3      8K   0.6s  low
  OK   qwen3-1.7b                   Q8_0       52.6-87.6     40K   0.7s  low
  OK   qwen2.5-3b                   Q8_0       26.7-44.4     32K   1.1s  low
  OK   phi-3.5-mini                 Q8_0       23.7-39.6     23K   1.4s  low
  OK   phi-4-mini                   Q8_0       23.6-39.3     70K   1.4s  low
  OK   qwen3-4b                     Q8_0       22.5-37.5     40K   1.5s  low

  showing 8 of 26 - ranked by verdict, then speed, then context (--all for every quant)
```

## Install

```sh
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/DEEPESH-845/ZeroCloud/main/install.sh | sh
```

Windows: download `zc-x86_64-pc-windows-msvc.exe` from
[Releases](https://github.com/DEEPESH-845/ZeroCloud/releases/latest), rename it
to `zc.exe`, and run it from a terminal.

From source (Rust 1.85+, edition 2024):

```sh
git clone https://github.com/DEEPESH-845/ZeroCloud && cd ZeroCloud
cargo build --release   # ./target/release/zc
```

The binary is under 5 MB, statically linked on Linux (musl), and has no runtime
dependencies. Nothing to install alongside it.

## Why not just look up your specs

Because spec sheets are wrong in exactly the ways that matter on cheap
hardware, and every one of them is invisible:

- single-channel RAM — half the bandwidth the spec implies, extremely common
- an iGPU quietly holding 1–2 GB of your system memory
- a DRAM-less SSD reading 4x slower than advertised
- WSL2 defaulting to half your host RAM
- thermal throttling that is already happening
- Windows Defender or FileVault sitting in the read path
- models stored in OneDrive or iCloud, where files are stubs that re-download

A 20-second benchmark catches all of them at once. `zc` measures RAM bandwidth
(STREAM triad, above cache), disk (O_DIRECT / F_NOCACHE random reads on *the
volume your models live on*), and compute (tiled f32 and int8 GEMM), then does
the memory-bound arithmetic:

```
t_token  = resident·bytes/BW_ram + (1−resident)·bytes/BW_disk
tok/s    = η(backend, quant) / t_token
max_ctx  = (usable − weights − compute_buffers) / kv_bytes_per_token
```

`η` is the one term that cannot be derived. It comes from real measured runs —
which is what `zc verify` collects and what `data/calibration/gate.jsonl` holds.

## Every number is checkable

The house rule, enforced throughout: **a number is measured, derived from
measured inputs, or printed as `-`.** No fallback constants. TTFT stays `-`
until a real run on your backend has been measured, because deriving it from a
CPU benchmark was measured to be wrong by 10–40x.

Predictions are ranges, never points, and each carries the confidence tier its
evidence earns. Ranges narrow as the dataset grows; they are never narrowed by
hand.

```sh
zc fit     # the coefficients, and how many real runs back each one
zc gate    # how wrong we have been, out of sample
zc verify  # run a real model for 30s: predicted vs actual, appended to your dataset
zc doctor  # everything probed and concluded, as Markdown for a bug report
```

**Current accuracy** — recompute it yourself with `zc gate` from a clean clone:

| | |
|---|---|
| median error, per machine | **9.6%** |
| machines | 6 (5 hypervisor, 1 bare metal), 8 runs |
| measurement landed inside the published range | 62.5% |

This is **pre-1.0, and the Phase 0 gate has not passed.** The gate is median
error under 25% across at least 5 machines *including 2 on bare metal*. The
median clears it with room to spare. The bare-metal count does not: there is
one, and it is the laptop this was written on.

That number is deliberately not something the author can fix alone — cloud
runners are hypervisors, and hypervisors run 10–30% below real hardware, so a
gate passed entirely in CI would prove nothing. **The missing evidence is one
real machine that is not this one.** If you have a Windows laptop, a Linux
desktop, or an old Intel Mac, that is the single most valuable thing anyone can
contribute right now, and it takes about twenty minutes.

One machine (a macOS VM) is currently missing by 410% and is not being hidden:
it is in the dataset, dragging the number down, until somebody understands why.
Ranges are wide on purpose while the evidence is thin, and they narrow only as
machines arrive — never by hand.

Adding a machine is the single most useful contribution right now, and it is
three commands:

```sh
ollama pull qwen3:1.7b     # or measure whatever you already have
zc verify qwen3:1.7b       # 30s: predicted vs actual, written to your disk only
zc share                   # shows you the record, then offers to open a browser
```

`zc share` prints the whole record first — every field, and the list of what is
*not* in it — then builds a GitHub URL with that record prefilled and asks
before opening it:

```
  hw                     8bc574063a10f63c
  os                     macos
  backend                Metal
  ...
  error_pct              4.9
  within_range           true

  not in it: hostname, username, serial number, MAC, IP, file paths

  lands at      data/calibration/community/8bc574063a10f63c-921a62a1.jsonl
  open in browser? [y/N]
```

`zc` still opens no connection — your browser does, and you watch it happen.
GitHub forks the repo to your account when you commit, a validator checks the
file, and a human merges it. There is no account to make and no token anywhere.

Merged records feed `zc fit` immediately, so your machine improves everyone's
predictions. They do not move the headline accuracy number above, which is
computed from `data/calibration/gate.jsonl` — the tier whose provenance is
known. `zc gate` prints both figures, always.

## Privacy

Zero network by default. `zc check`, `zc verify`, `zc fit`, `zc gate`,
`zc doctor` and `zc share` make no outbound connection of any kind; there is no telemetry and
no analytics. `zc verify` writes one line to `data/calibration/local.jsonl` on
your own disk and nowhere else.

`zc share` is the only command that sends anything anywhere, and it does not do
the sending: it prints the record, prints a URL containing it, and asks before
handing that URL to your browser. Nothing is uploaded by `zc` itself, so there
is no token to leak and nothing to trust us about that you cannot read on your
own terminal first.

Reports carry no hostname, username, serial number, MAC or IP, and paths are
rewritten to `~`.

## Contributing

The highest-value contributions need no Rust:

- **Add your machine.** `zc verify` then `zc share`. Old, slow and unusual
  hardware is worth more than another fast laptop.
- **Add a model.** One JSON file in `data/models/`. Nothing else to touch.
- **Report a bad prediction.** `zc verify` prints predicted vs actual; paste it
  with `zc doctor` output. A prediction that was wrong is data, not a
  complaint.

See [CONTRIBUTING.md](CONTRIBUTING.md).

## What this is not

It does not run models — Ollama, llama.cpp and LM Studio do that, and `zc`
reads their timings rather than replacing them. It does not phone home. It does
not rank models by quality; it reports bytes, bandwidth and speed, all of which
are measurable, and leaves taste to you.

## License

Apache-2.0. See [LICENSE](LICENSE).
