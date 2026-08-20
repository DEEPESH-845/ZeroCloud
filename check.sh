#!/usr/bin/env bash
# Everything that must pass before a commit.
#
# Cross-target `check` is not optional: zc-bench once failed to compile on
# Windows for days because nothing here caught it, and Windows is the primary
# target market. `cargo check` needs only the target's std, no linker.
set -euo pipefail

TARGETS=(aarch64-apple-darwin x86_64-unknown-linux-gnu x86_64-pc-windows-msvc)

echo "== tests =="
cargo test --workspace --quiet

echo "== clippy =="
cargo clippy --workspace --all-targets -- -D warnings

for t in "${TARGETS[@]}"; do
  if rustup target list --installed | grep -qx "$t"; then
    echo "== check $t =="
    # clippy, not check: the host clippy pass above never compiles the
    # `#[cfg(target_os = ...)]` blocks in disk.rs, cpu.rs and gpu.rs, so a lint
    # error inside them stayed invisible locally and turned CI red for days.
    # clippy subsumes check, so this is strictly stronger and no slower.
    cargo clippy --workspace --all-targets --target "$t" --quiet -- -D warnings
  else
    echo "== skip $t (not installed: rustup target add $t) =="
  fi
done

echo "== calibration =="
# The same script CI runs on every pull request touching crates/zc-model/data/calibration.
# Self-test first: a validator whose own fixtures fail would fail open.
python3 scripts/validate_calibration.py --self-test
python3 scripts/validate_calibration.py

echo "== installer =="
sh -n install.sh
# Offline: ZC_VERSION skips the network tag lookup, so this exercises target
# detection and URL construction without needing a published release. The asset
# names here must match .github/workflows/release.yml exactly -- a mismatch
# means every `curl | sh` 404s, and nothing else would catch it.
case "$(uname -s)/$(uname -m)" in
  Darwin/arm64)  want=aarch64-apple-darwin ;;
  Darwin/x86_64) want=x86_64-apple-darwin ;;
  Linux/x86_64)  want=x86_64-unknown-linux-musl ;;
  Linux/aarch64) want=aarch64-unknown-linux-musl ;;
  *)             want=UNSUPPORTED ;;
esac
ZC_VERSION=v0.0.0 ZC_DRY_RUN=1 sh install.sh | grep -q "zc-$want" \
  || { echo "installer resolved the wrong target (expected zc-$want)"; exit 1; }
echo "resolves zc-$want"
# With only prereleases published, /releases/latest redirects to /releases and
# the tag parser passed the whole URL through -- non-empty, so the old
# emptiness check let it build a download URL with an https:// in the middle.
# A tag never contains a slash.
if ZC_VERSION='https://github.com/x/releases' ZC_DRY_RUN=1 sh install.sh >/dev/null 2>&1; then
  echo "installer accepted a URL as a release tag"; exit 1
fi
echo "rejects a URL-shaped tag"

echo "== contracts =="
# Every promise zc makes to something that is not a human at a terminal:
# no escape sequences in a pipe, silent stderr, valid JSON, scoped flags,
# 80 columns, no account name. Portable so CI runs the identical set on
# Windows, where this file cannot.
python3 scripts/contract_smoke.py ./target/release/zc

echo "== signals =="
# Drop guards do not run when a signal kills the process, and the disk
# benchmark holds a 512 MiB scratch file through the slowest phase of a run --
# which is when an impatient user presses Ctrl-C. Reproduced before it was
# fixed; this keeps it fixed.
python3 scripts/signal_smoke.py ./target/release/zc

echo "== tui =="
# The unit tests cover the state machine and the frame without a terminal.
# They cannot cover whether the binary opens the TUI at all or whether the data
# it hands over is the data the keys expect -- which is how `a` shipped as an
# advertised key that did nothing. Skips cleanly where there is no pty.
python3 scripts/tui_smoke.py ./target/release/zc

echo "OK"
