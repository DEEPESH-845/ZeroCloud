#!/usr/bin/env python3
"""What `zc` leaves behind when it is interrupted.

`Drop` guards are the codebase's cleanup mechanism and they are correct for
normal returns and for `?` paths. They do not run when a signal terminates the
process, because the process does not unwind — and the disk benchmark holds a
**512 MiB** scratch file open for the slowest phase of the run, which is
exactly when an impatient user presses Ctrl-C.

The leak this guards was reproduced before it was fixed: interrupt during the
benchmark and `.zc-bench-scratch.tmp` stays in the model directory, hidden by
its leading dot, on the volume a low-end laptop has least room on.

A pty is needed because a background job inherits `SIG_IGN` for SIGINT from a
non-interactive shell, so the signal never arrives. Skips cleanly where no pty
can be allocated.
"""

import os
import pty
import select
import signal
import subprocess
import sys
import tempfile
import time

ZC = sys.argv[1] if len(sys.argv) > 1 else "target/release/zc"
if not os.path.exists(ZC) and os.path.exists(ZC + ".exe"):
    ZC += ".exe"

SCRATCH_NAME = ".zc-bench-scratch.tmp"
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main():
    if not os.path.exists(ZC):
        print(f"signal_smoke: {ZC} not built, skipping")
        return 0
    if not hasattr(signal, "SIGINT") or os.name != "posix":
        print("signal_smoke: POSIX signals only, skipping")
        return 0

    home = tempfile.mkdtemp(prefix="zc-signal-")
    scratch = os.path.join(home, SCRATCH_NAME)
    env = {
        **os.environ,
        "HOME": home,
        "XDG_DATA_HOME": home,
        "TERM": "xterm-256color",
        "LANG": "en_US.UTF-8",
    }
    try:
        m, s = pty.openpty()
    except OSError as e:
        print(f"signal_smoke: cannot allocate a pty ({e}), skipping")
        return 0

    # --no-tui so the run is the benchmark and nothing else, and
    # start_new_session so the child leads a process group we can signal the
    # way a terminal does.
    p = subprocess.Popen(
        [os.path.abspath(ZC), "check", "--no-tui", "--top", "1"],
        stdin=s,
        stdout=s,
        stderr=s,
        cwd=home,
        env=env,
        start_new_session=True,
    )
    os.close(s)

    sent = False
    size = 0
    deadline = time.time() + 120
    while time.time() < deadline and p.poll() is None:
        if not sent and os.path.exists(scratch):
            try:
                size = os.path.getsize(scratch)
            except OSError:
                size = 0
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
            sent = True
        r, _, _ = select.select([m], [], [], 0.02)
        if r:
            try:
                if not os.read(m, 65536):
                    break
            except OSError:
                break
    try:
        rc = p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        p.kill()
        rc = "timeout"
    os.close(m)

    if not sent:
        # An existing model file was used, so no scratch file was ever created.
        # Nothing to leak and nothing to assert.
        print("signal_smoke: no scratch file was needed on this machine, skipping")
        _rmtree(home)
        return 0

    check("the scratch file was really created", size > 0, f"{size / 2**20:.0f} MiB")
    time.sleep(0.5)
    leaked = os.path.exists(scratch)
    check(
        "an interrupted run leaves no scratch file behind",
        not leaked,
        f"{os.path.getsize(scratch) / 2**20:.0f} MiB left" if leaked else "",
    )
    # 130 is the shell's convention for "killed by SIGINT"; Python reports the
    # negative signal number. Either says the signal was honoured rather than
    # swallowed, which is what a caller in a script needs to see.
    check(
        "the exit status reports the signal",
        rc in (-signal.SIGINT, 130),
        f"exit={rc}",
    )

    _rmtree(home)
    if FAILURES:
        print(f"\nsignal_smoke: {len(FAILURES)} failure(s): {', '.join(FAILURES)}")
        return 1
    print("signal_smoke: ok")
    return 0


def _rmtree(path):
    try:
        import shutil

        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
