#!/usr/bin/env python3
"""Drive the TUI on a real pty and assert it behaves.

The unit tests in `zc-tui` cover the state machine and the frame renderer
without a terminal, which is most of the logic. They cannot cover the part that
only exists once a terminal is attached: whether the binary opens the TUI at
all, whether the data it hands over is the data the keys expect, and whether it
gives the terminal back on the way out.

That gap shipped a real bug. `a` (show every quantisation) was advertised in the
footer and in the help overlay while doing nothing, because `check.rs` collapsed
the model set to one row per model *before* the TUI ever saw it. Every unit test
passed, clippy was clean, and the only way to notice was to press the key.

`script` does not work in a sandbox, so this uses `pty.openpty` directly.
Skips cleanly where a pty cannot be allocated, because a CI runner without one
should not fail the build.
"""

import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import time

try:
    import fcntl
    import termios
except ImportError:  # Windows
    print("tui_smoke: no pty on this platform, skipping")
    sys.exit(0)

ZC = sys.argv[1] if len(sys.argv) > 1 else "target/release/zc"
# The benchmark runs before the first frame. Generous, because the disk probe
# is the slow part and a loaded CI runner is slower still.
BOOT = float(os.environ.get("ZC_SMOKE_BOOT", "25"))
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


class Tui:
    def __init__(self, rows=24, cols=90, env=None):
        self.m, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        e = {**os.environ, "TERM": "xterm-256color", "LANG": "en_US.UTF-8"}
        e.pop("ZC_ASCII", None)
        if env:
            e.update(env)
        self.p = subprocess.Popen(
            [os.path.abspath(ZC)],
            stdin=slave,
            stdout=slave,
            stderr=subprocess.DEVNULL,
            env=e,
            cwd="/tmp",
        )
        os.close(slave)

    def pump(self, seconds, until=None):
        """Read for `seconds`, or until `until(text)` is true.

        The early exit is what keeps this usable in `check.sh`: waiting out the
        full benchmark budget on every step turned a two-second script into a
        minute of nothing.
        """
        buf = b""
        end = time.time() + seconds
        while time.time() < end:
            r, _, _ = select.select([self.m], [], [], 0.2)
            if r:
                try:
                    d = os.read(self.m, 65536)
                except OSError:
                    break
                if not d:
                    break
                buf += d
                if until is not None and until(buf.decode(errors="replace")):
                    break
            elif self.p.poll() is not None:
                break
        return buf.decode(errors="replace")

    def send(self, keys, wait=1.5, until=None):
        os.write(self.m, keys)
        return self.pump(wait, until)

    def close(self):
        if self.p.poll() is None:
            self.p.kill()
        try:
            os.close(self.m)
        except OSError:
            pass


def footer_count(text):
    """The `N of M` in the footer, as a pair, or None."""
    plain = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "\n", text)
    hits = re.findall(r"(\d+) of (\d+) ", plain)
    return (int(hits[-1][0]), int(hits[-1][1])) if hits else None


def main():
    if not os.path.exists(ZC):
        print(f"tui_smoke: {ZC} not built, skipping")
        return 0
    try:
        t = Tui()
    except OSError as e:
        print(f"tui_smoke: cannot allocate a pty ({e}), skipping")
        return 0

    boot = t.pump(BOOT, until=lambda x: footer_count(x) is not None)
    check("progress shown while benchmarking", "ram" in boot and "disk" in boot)
    check("the table opens on a terminal", footer_count(boot) is not None)

    base = footer_count(boot)
    if base is None:
        t.close()
        print("tui_smoke: no table appeared; nothing further can be checked")
        return 1

    # `a` must actually change the row set. This is the bug this file exists
    # for: the key was advertised and inert.
    seen = lambda want: (lambda x: footer_count(x) not in (None, want))
    after_a = footer_count(t.send(b"a", until=seen(base)))
    check(
        "`a` expands to every quantisation",
        after_a is not None and after_a[1] > base[1],
        f"{base[1]} -> {after_a[1] if after_a else '?'}",
    )
    back = footer_count(t.send(b"a", until=seen(after_a)))
    check("`a` toggles back", back == base, f"{back} vs {base}")

    # Filtering narrows, and the denominator follows the view.
    filtered = footer_count(t.send(b"/qwen3\r", until=seen(base)))
    check(
        "filter narrows the rows",
        filtered is not None and filtered[0] < base[0] and filtered[1] == base[1],
        str(filtered),
    )

    # Esc clears a committed filter instead of exiting.
    cleared = footer_count(t.send(b"\x1b", until=seen(filtered)))
    check("esc clears a committed filter", cleared == base, str(cleared))
    check("esc did not exit", t.p.poll() is None)

    # The detail pane explains the selected row.
    pane = t.send(b"\r", until=lambda x: "bandwidth" in x)
    check(
        "enter shows the derivation",
        all(w in pane for w in ("weights", "bandwidth", "eta", "context")),
    )
    t.send(b"\r")

    # The help overlay lists the keys.
    helped = t.send(b"?", until=lambda x: "filter by name" in x)
    check("? lists the keys", "filter by name" in helped)
    t.send(b"\x1b")

    # Resize must reflow, never overrun. SIGWINCH is sent explicitly: setting
    # the window size on the master only signals the pty's foreground process
    # group, which the child is not reliably in, and without it this check
    # measures an empty capture and passes for the wrong reason.
    fcntl.ioctl(t.m, termios.TIOCSWINSZ, struct.pack("HHHH", 14, 54, 0, 0))
    t.p.send_signal(signal.SIGWINCH)
    # A resize alone triggers a redraw, but nudge the cursor so there is
    # certain to be a frame to measure.
    resized = t.send(b"j", wait=2.0)
    lines = [
        l.rstrip()
        for l in re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "\n", resized).split("\n")
        if l.strip()
    ]
    widest = max((len(l) for l in lines), default=0)
    check(
        "the resize produced a frame to measure",
        widest > 0,
        f"{len(lines)} lines",
    )
    check("reflows on resize without overrunning", 0 < widest <= 54, f"widest {widest}")

    # Quitting restores the shell and leaves the answer behind.
    # No early exit here: the child writes the whole static report on the way
    # out, and a reader that stops draining the pty leaves it blocked in
    # write() forever. Drain until it actually exits -- `pump` returns as soon
    # as output stops and the process is gone.
    tail = t.send(b"q", wait=10.0)
    try:
        rc = t.p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        rc = "TIMEOUT"
    check("q exits 0", rc == 0, f"exit={rc}")
    check("the report is left in the scrollback", "== predictions ==" in tail)
    check("the alternate screen is released", "?1049l" in tail)
    t.close()

    # Too small to draw: say so rather than paint something unreadable.
    small = Tui(rows=9, cols=39)
    s = small.pump(BOOT, until=lambda x: "too small" in x)
    check("a tiny terminal says so", "too small" in s)
    small.send(b"q", wait=2.0)
    small.close()

    if FAILURES:
        print(f"\ntui_smoke: {len(FAILURES)} failure(s): {', '.join(FAILURES)}")
        return 1
    print("tui_smoke: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
