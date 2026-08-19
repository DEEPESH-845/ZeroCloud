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
# The same script CI runs on every pull request touching data/calibration.
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

echo "== non-tty output =="
# The TUI is default-on for a human. Everything else -- a pipe, a redirect,
# --json, CI, an agent -- must still get the plain renderer. These run piped,
# so if a TUI escape sequence ever reaches stdout it fails right here.
cargo build --release --quiet
if ./target/release/zc check --top 3 2>/dev/null | grep -q "$(printf '\033')"; then
  echo "escape sequences leaked into piped stdout"; exit 1
fi
./target/release/zc check --json 2>/dev/null | python3 -m json.tool >/dev/null \
  || { echo "--json is not valid JSON"; exit 1; }
# Progress writes to stderr and only on a terminal; piped it must be silent.
if [ -s /dev/stdin ] 2>/dev/null; then :; fi
if [ "$(./target/release/zc check --top 1 2>&1 >/dev/null | wc -c)" -ne 0 ]; then
  echo "progress wrote to a non-tty stderr"; exit 1
fi
# --tui with nowhere to draw is an error, never a silent downgrade.
if ./target/release/zc check --tui </dev/null >/dev/null 2>&1; then
  echo "--tui succeeded without a terminal"; exit 1
fi
# No line of `zc check` may exceed 80 columns -- the table wrapped hardest on
# the low-end machines this tool is for.
if ./target/release/zc check --all 2>/dev/null | awk 'length>80' | grep -q .; then
  echo "a row ran past 80 columns"; exit 1
fi
# No line of terminal output may exceed 80 columns. `zc doctor` is exempt --
# it is Markdown for a GitHub issue, where soft wrap is correct -- and so is
# the share URL, which must stay one copy-pasteable token.
for c in "check" "check --all" "fit" "gate" "--help"; do
  # shellcheck disable=SC2086
  if ./target/release/zc $c 2>/dev/null | awk 'length>80' | grep -q .; then
    echo "a line of \`zc $c\` ran past 80 columns"; exit 1
  fi
done
echo "plain when piped, silent stderr, 80 columns"

echo "== no account name in any output =="
# `zc doctor` is documented as paste-into-a-public-issue and `--json` gets
# attached to bug reports, so no surface may print the user's home path. Run
# from a temp directory: inside the repo the calibration file resolves to a
# relative path and the leak would hide.
tmp=$(mktemp -d)
zcbin="$PWD/target/release/zc"
leak=0
for c in "check" "check --json" "doctor" "fit" "gate" "share --print"; do
  # shellcheck disable=SC2086
  if (cd "$tmp" && "$zcbin" $c 2>/dev/null) | grep -qF "$HOME"; then
    echo "\`zc $c\` printed \$HOME"; leak=1
  fi
done
rm -rf "$tmp"
[ "$leak" -eq 0 ] || exit 1
echo "no \$HOME in check, json, doctor, fit, gate, share"

echo "OK"
