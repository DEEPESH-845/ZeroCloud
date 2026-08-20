<!--
Two questions. Everything else about this project is negotiable; these are not.
-->

## Did `./check.sh` pass?

- [ ] Yes

It runs the tests, clippy, the cross-compile for Linux and Windows, the
calibration validator, the installer self-check, and the three smoke scripts
(contracts, TUI, signals). CI runs the same set on three operating systems, so
this is the fastest way to find out before it does.

## Is every number here measured, or derived from something measured?

- [ ] Yes
- [ ] This change adds no numbers

The one rule this project does not bend: a number is measured, derived from
measured inputs, or printed as `-`. Not a plausible constant, not a spec-sheet
lookup, not an average of what similar hardware usually does. If a value cannot
be measured, say so in the output rather than substituting one — a missing
number is recoverable, a confidently wrong one is not.

If you are adding a calibration record, `scripts/validate_calibration.py`
recomputes `error_pct` and `within_range` from the record's own inputs, so a
record that disagrees with itself fails CI.

## What is this?

<!-- A sentence is fine. If it fixes something, what was the symptom? -->
