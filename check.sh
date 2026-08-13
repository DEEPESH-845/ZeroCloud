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
    cargo check --workspace --all-targets --target "$t" --quiet
  else
    echo "== skip $t (not installed: rustup target add $t) =="
  fi
done

echo "OK"
