#!/usr/bin/env python3
"""The promises `zc` makes to anything that is not a human at a terminal.

`check.sh` enforces these on the maintainer's machine, but it is bash and does
not run on Windows — which is the primary target market and the platform whose
code paths `VERIFICATION.md` records as least exercised. This is the same set
of checks, portable, so CI can run them on ubuntu, macos and windows alike.

Everything here works without a terminal, which is the point: these are exactly
the promises that hold when stdout is a pipe, a file, a CI log or an agent.

What this deliberately does NOT cover is the interactive TUI — raw mode, key
events, resize. That needs a pty, which `scripts/tui_smoke.py` provides on Unix
and which has no portable equivalent here. Writing a Windows ConPTY driver
blind would be the same mistake the crossterm dependency exists to avoid, so
the gap is recorded in VERIFICATION.md rather than papered over.
"""

import json
import os
import subprocess
import sys

ZC = sys.argv[1] if len(sys.argv) > 1 else "target/release/zc"
# Windows names the binary zc.exe; accept either so CI can pass one path.
if not os.path.exists(ZC) and os.path.exists(ZC + ".exe"):
    ZC += ".exe"
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def run(*args, cwd=None):
    """Run zc with stdout and stderr captured — so never a terminal."""
    p = subprocess.run(
        [os.path.abspath(ZC), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=600,
    )
    return p.returncode, p.stdout, p.stderr


def main():
    if not os.path.exists(ZC):
        print(f"contract_smoke: {ZC} not built")
        return 1

    # -- the TUI must never reach a pipe -------------------------------------
    rc, out, err = run("check", "--top", "3")
    check("check exits 0 when piped", rc == 0, f"exit={rc}")
    check("no escape sequences in piped stdout", "\x1b" not in out)
    check("progress is silent when stderr is not a terminal", err == "", repr(err[:40]))

    # -- --json stays machine-readable ---------------------------------------
    rc, out, err = run("check", "--json")
    check("--json exits 0", rc == 0)
    try:
        doc = json.loads(out)
        ok = isinstance(doc.get("schema"), int) and len(doc.get("models", [])) > 0
        check("--json is valid JSON with models", ok, f"schema={doc.get('schema')}")
        paths = [
            doc["machine"]["storage"]["model_dir"],
            doc["machine"]["storage"]["mount"],
        ]
        home = os.path.expanduser("~")
        check(
            "--json carries no account name",
            all(home not in p for p in paths),
            str(paths),
        )
        # An unmeasured value must be null, never a substituted number.
        rows = doc["models"]
        ok = all(("ttft_s" in r) for r in rows)
        check("every model row states ttft_s (null when unmeasured)", ok)
    except (ValueError, KeyError) as e:
        check("--json is valid JSON with models", False, str(e))

    # -- flags are scoped to the commands that read them ----------------------
    for cmd, flag in [
        ("doctor", "--json"),
        ("doctor", "--tui"),
        ("fit", "--kv"),
        ("gate", "--json"),
        ("share", "--top"),
        ("plan", "--json"),
        ("check", "--target-tps"),
    ]:
        val = {"--kv": ["f16"], "--top": ["3"], "--target-tps": ["5"]}.get(flag, [])
        args = [cmd, flag] + val
        rc, _, err = run(*args)
        check(f"`zc {cmd} {flag}` is refused", rc == 2, f"exit={rc}")

    # -- a requested TUI with nowhere to draw is an error, not a downgrade ----
    rc, out, err = run("check", "--tui")
    check("--tui without a terminal exits 2", rc == 2, f"exit={rc}")

    # -- a typo suggests the real command ------------------------------------
    rc, _, err = run("chekc")
    check("a near-miss command suggests the real one", rc == 2 and "check" in err)

    # -- 80 columns, on every human-facing surface ---------------------------
    for args in (
        ["check", "--all"],
        ["check"],
        ["--help"],
        ["fit"],
        ["gate"],
        ["plan", "qwen3-8b", "--context", "32K"],
    ):
        rc, out, _ = run(*args)
        wide = [l for l in out.splitlines() if len(l) > 80]
        check(f"`zc {' '.join(args)}` fits 80 columns", not wide, f"{len(wide)} over")

    # -- no surface prints the user's home path ------------------------------
    # Run from elsewhere: inside a checkout the dataset path is relative and a
    # leak would hide.
    home = os.path.expanduser("~")
    tmp = os.path.dirname(os.path.abspath(ZC))
    for args in (
        ["check", "--top", "1"],
        ["doctor"],
        ["fit"],
        ["gate"],
        ["share", "--print"],
    ):
        _, out, _ = run(*args, cwd=tmp)
        check(f"`zc {' '.join(args)}` prints no $HOME", home not in out)

    # -- the dataset is compiled in, so an installed user is not on priors ----
    rc, out, _ = run("fit", cwd=tmp)
    # Normalise whitespace: the header is wrapped to 80 columns, so the phrase
    # is routinely split across two lines.
    flat = " ".join(out.split())
    check("the calibration dataset is embedded", "shipped in this binary" in flat)

    serve_http()
    serve_mcp()

    if FAILURES:
        print(f"\ncontract_smoke: {len(FAILURES)} failure(s): {', '.join(FAILURES)}")
        return 1
    print("contract_smoke: ok")
    return 0


def serve_http():
    """`zc serve` on an ephemeral port: the routes, and the two refusals that
    are its security model."""
    import http.client
    import re
    import time

    p = subprocess.Popen(
        [os.path.abspath(ZC), "serve", "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        port = None
        deadline = time.time() + 600
        while time.time() < deadline and p.poll() is None:
            line = p.stderr.readline()
            m = re.search(r"127\.0\.0\.1:(\d+)", line)
            if m:
                port = int(m.group(1))
                break
        check("serve announces its port on stderr", port is not None)
        if port is None:
            return

        def get(path, method="GET", host=None):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            c.putrequest(method, path, skip_host=host is not None)
            if host is not None:
                c.putheader("Host", host)
            c.endheaders()
            r = c.getresponse()
            body = r.read().decode()
            c.close()
            return r.status, body

        status, body = get("/v1/check?top=2")
        doc = json.loads(body) if status == 200 else {}
        check("serve /v1/check is the check --json document", len(doc.get("models", [])) == 2, f"status={status}")
        status, body = get("/v1/plan?model=qwen3-0.6b")
        check("serve /v1/plan returns a plan", status == 200 and json.loads(body).get("model") == "qwen3-0.6b", f"status={status}")
        check("serve refuses an unknown parameter", get("/v1/check?al_quants=1")[0] == 400)
        check("serve refuses a rebound Host", get("/health", host="evil.example")[0] == 403)
        check("serve refuses anything but GET", get("/v1/check", method="POST")[0] == 405)
    finally:
        p.kill()
        out, _ = p.communicate()
    check("serve writes nothing to stdout", out == "", repr(out[:40]))


def serve_mcp():
    """`zc serve --mcp`: a handshake, a tool list and one real tool call."""
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                    "clientInfo": {"name": "contract_smoke", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "check", "arguments": {"top": 1}}},
    ]
    p = subprocess.run(
        [os.path.abspath(ZC), "serve", "--mcp"],
        input="".join(json.dumps(m) + "\n" for m in msgs),
        capture_output=True, text=True, timeout=600,
    )
    try:
        replies = {r["id"]: r for r in map(json.loads, p.stdout.splitlines())}
    except ValueError:
        replies = {}
    check("mcp answers every request and no notification", sorted(replies) == [1, 2, 3], repr(p.stdout[:80]))
    if sorted(replies) != [1, 2, 3]:
        return
    check("mcp negotiates the requested version", replies[1]["result"]["protocolVersion"] == "2025-11-25")
    tools = [t["name"] for t in replies[2]["result"]["tools"]]
    check("mcp lists check and plan", tools == ["check", "plan"], repr(tools))
    res = replies[3]["result"]
    doc = json.loads(res["content"][0]["text"]) if not res.get("isError") else {}
    check("mcp check returns the check --json document", len(doc.get("models", [])) == 1)


if __name__ == "__main__":
    sys.exit(main())
