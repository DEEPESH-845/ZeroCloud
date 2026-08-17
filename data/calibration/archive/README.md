# Archived calibration records

Records here are **history, not evidence**. `zc gate` and `zc fit` read a single
file (`gate.jsonl`, else `local.jsonl`); nothing in this directory is ever read.

A record is archived only for a stated reason, and "it made the number worse" is
never one of them.

## 2026-08-16-pre-coverage-factor.jsonl

One run: `qwen3:4b` Q4_K_M on Apple Silicon, -28.3% error, outside its published
range.

Retired for two reasons:

1. **Unknown provenance.** It predates the `virt` field. `zc gate` correctly
   refuses to assume an unlabelled record came from bare metal, so its presence
   marked that machine virtualized forever and barred it from `MIN_BARE_METAL`.
2. **Superseded grader.** Its `error_pct` grades a prediction made before the
   1.645σ coverage factor, the 0.40 prior floor, and the f16 KV default. The
   gate reads `error_pct` from the record rather than recomputing it — that is
   what keeps errors genuinely out-of-sample — so a record cannot be re-graded
   in place. It can only be retired and re-measured.

The machine it came from is re-measured in `gate.jsonl` under the current model.
