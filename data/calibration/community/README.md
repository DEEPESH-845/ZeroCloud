# Community calibration records

One file per submitted `zc verify` run, named `<hw>-<hash8>.jsonl`, holding
exactly one record on one line. `zc share` produces both the name and the
content; `scripts/validate_calibration.py` checks every file here on every
pull request.

These records feed `zc fit`, so a merged submission improves the coefficients
behind everyone's predictions. They do **not** move the headline accuracy
figure `zc gate` prints, which is computed from `../gate.jsonl` — the tier
whose provenance we can state. `zc gate` prints both numbers.

Promotion from here into `gate.jsonl` is a deliberate `git mv` with a human's
name on the commit. There is no tooling for it on purpose.
