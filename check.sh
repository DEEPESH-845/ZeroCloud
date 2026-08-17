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

echo "OK"
