#!/usr/bin/env python3
"""Validate every calibration record in the repository.

Runs over the whole tree rather than only the files a pull request touched:
repo-wide integrity is the invariant, and a change that corrupts an untouched
file is precisely what a changed-files-only validator misses.

The important checks are not schema checks. A record carries the inputs its
own conclusions were computed from, so `error_pct` and `within_range` are both
recomputable here -- which means a fabricated "0% error" submission has to
contradict itself to exist. It cannot stop a consistent lie from a machine
nobody can inspect; that is what the two provenance tiers and a human on the
merge button are for.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "crates" / "zc-model" / "data" / "calibration"
NAME_RE = re.compile(r"^[0-9a-f]{16}-[0-9a-f]{8}\.jsonl$")
MAX_BYTES = 4096
VIRT = {"none", "hypervisor", "wsl2", "container"}

REQUIRED = [
    "hw", "os", "virt", "backend", "runtime", "ram_bw_gbs", "vram_bw_gbs",
    "disk_bw_gbs", "gflops", "threads", "kv", "model", "quant", "ctx",
    "prompt_tokens", "eval_tokens", "predicted_lo", "predicted_hi",
    "actual_decode_tok_s", "actual_prefill_tok_s", "assumed_eta",
    "implied_eta", "implied_prefill_scale", "active_params", "error_pct",
    "within_range",
]
ALLOWED = set(REQUIRED)

# Bounds a schema cannot express. Wide on purpose: the job is to reject the
# physically impossible, not to have an opinion about what hardware exists.
RANGES = {
    "ram_bw_gbs": (1.0, 2000.0),
    "gflops": (0.0, 1e6),
    "threads": (1, 1024),
    "ctx": (1, 10_000_000),
    "implied_eta": (0.0, 1.5),
}


def fnv1a(s):
    """FNV-1a-64, byte-identical to `zc_runtime::calibrate::fingerprint`."""
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def check_record(rec, where, out):
    missing = [k for k in REQUIRED if k not in rec]
    for key in missing:
        out.append(f"{where}: missing required field {key!r}")
    # Every check below indexes fields directly, so a record missing any of
    # them is reported and abandoned rather than crashing the whole run.
    if missing:
        return
    extra = set(rec) - ALLOWED
    if extra:
        out.append(f"{where}: unknown field(s) {sorted(extra)}")
    if rec["virt"] not in VIRT:
        out.append(f"{where}: virt {rec['virt']!r} not one of {sorted(VIRT)}")
    for key, (lo, hi) in RANGES.items():
        v = rec[key]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not (lo < v <= hi):
            out.append(f"{where}: {key}={v} outside ({lo}, {hi}]")
    lo, hi, act = rec["predicted_lo"], rec["predicted_hi"], rec["actual_decode_tok_s"]
    if not (0 < lo <= hi):
        out.append(f"{where}: predicted range {lo}-{hi} is not ordered and positive")
    if act <= 0:
        out.append(f"{where}: actual_decode_tok_s={act} is not positive")
        return
    # The two claims the record makes about itself, recomputed from the inputs
    # it carries. `zc verify` rounds error to one decimal place, hence 0.15.
    mid = (lo + hi) / 2
    want_err = (mid - act) / act * 100
    if abs(want_err - rec["error_pct"]) > 0.15:
        out.append(
            f"{where}: error_pct={rec['error_pct']} but its own numbers give {want_err:.1f}"
        )
    want_in = lo <= act <= hi
    if want_in != rec["within_range"]:
        out.append(
            f"{where}: within_range={rec['within_range']} but {act} vs {lo}-{hi} gives {want_in}"
        )


def check_community_file(path, out, curated_lines=frozenset()):
    where = path.name
    raw = path.read_bytes()
    if len(raw) > MAX_BYTES:
        out.append(f"{where}: {len(raw)} bytes, over the {MAX_BYTES} limit")
        return
    if not NAME_RE.match(where):
        out.append(f"{where}: filename must be <16-hex>-<8-hex>.jsonl")
        return
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        out.append(f"{where}: not valid UTF-8 ({e})")
        return
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) != 1:
        out.append(f"{where}: holds {len(lines)} records, must hold exactly 1")
        return
    line = lines[0].strip()
    try:
        rec = json.loads(line)
    except json.JSONDecodeError as e:
        out.append(f"{where}: not valid JSON ({e})")
        return
    if not isinstance(rec, dict):
        out.append(f"{where}: top level is {type(rec).__name__}, must be an object")
        return
    hw, digest = where[: -len(".jsonl")].split("-")
    if rec.get("hw") != hw:
        out.append(f"{where}: filename says hw={hw}, record says {rec.get('hw')!r}")
    if fnv1a(line)[:8] != digest:
        out.append(f"{where}: filename hash {digest} does not match its contents")
    # The readers dedupe, so a copy of a curated record changes no number -- but
    # it is also not evidence, and merging one silently would tell a contributor
    # their run landed when nothing was added.
    if line in curated_lines:
        out.append(f"{where}: byte-identical to a record already in gate.jsonl")
    check_record(rec, where, out)


def validate():
    out = []
    curated_lines = set()
    curated = CALIB / "gate.jsonl"
    if curated.is_file():
        for i, line in enumerate(curated.read_text().splitlines(), 1):
            if not line.strip():
                continue
            curated_lines.add(line.strip())
            try:
                check_record(json.loads(line), f"gate.jsonl:{i}", out)
            except json.JSONDecodeError as e:
                out.append(f"gate.jsonl:{i}: not valid JSON ({e})")
    community = CALIB / "community"
    if community.is_dir():
        for p in sorted(community.iterdir()):
            if p.suffix == ".jsonl":
                check_community_file(p, out, curated_lines)
    return out


GOOD = {
    "hw": "8bc574063a10f63c", "os": "macos", "virt": "none", "backend": "Metal",
    "runtime": "ollama", "ram_bw_gbs": 126.55, "vram_bw_gbs": 0.0,
    "disk_bw_gbs": 5.01, "gflops": 427.6, "threads": 4, "kv": "f16",
    "model": "qwen3:1.7b", "quant": "Q4_K_M", "ctx": 4096,
    "prompt_tokens": 979, "eval_tokens": 128, "predicted_lo": 60.0,
    "predicted_hi": 100.0, "actual_decode_tok_s": 80.0,
    "actual_prefill_tok_s": 2750.43, "assumed_eta": 0.616,
    "implied_eta": 0.922, "implied_prefill_scale": 26.1399,
    "active_params": 2031739904, "error_pct": 0.0, "within_range": True,
}


def self_test():
    """Hand-computed: the midpoint of 60-100 is 80 and the actual is 80, so the
    error is exactly 0.0% and the measurement is inside the range. Every fixture
    below perturbs one field of that and must be caught."""
    failures = []

    def case(name, rec, should_pass):
        out = []
        check_record(dict(rec), "fixture", out)
        if (not out) != should_pass:
            failures.append(f"{name}: expected {'pass' if should_pass else 'fail'}, got {out}")

    case("a correct record", GOOD, True)
    case("error_pct contradicts its own inputs", {**GOOD, "actual_decode_tok_s": 40.0}, False)
    case("within_range contradicts its own bounds", {**GOOD, "within_range": False}, False)
    case("impossible eta", {**GOOD, "implied_eta": 2.0}, False)
    case("unknown virt", {**GOOD, "virt": "xen"}, False)
    case("smuggled field", {**GOOD, "note": "trust me"}, False)
    case("missing error_pct", {k: v for k, v in GOOD.items() if k != "error_pct"}, False)

    # Filename-level fixtures need real files, written outside the repo tree.
    import tempfile

    line = json.dumps(GOOD, separators=(",", ":"))
    good_name = f"{GOOD['hw']}-{fnv1a(line)[:8]}.jsonl"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name, content, curated, should_pass in [
            (good_name, line + "\n", frozenset(), True),
            # The web editor's trailing newline is the contributor's browser's
            # choice, so its absence must not be an error.
            (good_name, line, frozenset(), True),
            (good_name, line + "\n" + line + "\n", frozenset(), False),
            (f"{GOOD['hw']}-deadbeef.jsonl", line + "\n", frozenset(), False),
            ("not-a-record.jsonl", line + "\n", frozenset(), False),
            (good_name, line + "\n", frozenset({line}), False),
        ]:
            p = d / name
            p.write_text(content)
            out = []
            check_community_file(p, out, curated)
            if (not out) != should_pass:
                failures.append(
                    f"{name} (curated={bool(curated)}): "
                    f"expected {'pass' if should_pass else 'fail'}, got {out}"
                )
            p.unlink()

    for f in failures:
        print(f"SELF-TEST FAIL  {f}")
    print(f"self-test: {'ok' if not failures else str(len(failures)) + ' failure(s)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    problems = validate()
    for p in problems:
        print(f"INVALID  {p}")
    print(f"{'FAILED' if problems else 'ok'}: {len(problems)} problem(s) in {CALIB}")
    sys.exit(1 if problems else 0)
