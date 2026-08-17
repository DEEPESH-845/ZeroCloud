# Gate Closure and Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `zc gate` green on ≥5 real machines, then ship ZeroCloud publicly as downloadable binaries with a README whose first number is the measured out-of-sample accuracy.

**Architecture:** The release pipeline is built *before* the gate campaign, so GitHub's native Windows/Linux/macOS runners replace local cross-compilation entirely and every calibration record is produced by the exact artifact users download. The stale single record is archived rather than deleted, a committed cross-machine dataset (`gate.jsonl`) becomes the file `zc fit` and `zc gate` read, and the field campaign appends to it.

**Tech Stack:** Rust 2024 edition, workspace of 6 crates, zero non-target-gated dependencies. GitHub Actions with the preinstalled `gh` CLI (no third-party actions). POSIX `sh` for the installer.

**Spec:** `docs/superpowers/specs/2026-08-17-gate-closure-and-launch-design.md`

## Global Constraints

- **A number is measured, derived from measured inputs, or printed as `-`.** No fallback constants, ever.
- Coefficients move by `zc fit` from calibration records, **never** tuned by hand — including when the gate is red.
- `zc-probe` / `zc-bench` / `zc-model` stay dependency-free. No dev-dependencies either; tests use `std` only.
- `./check.sh` must be green before every commit and before any tag is pushed.
- Repository is `DEEPESH-845/ZeroCloud`. Binary name is `zc`. Workspace version is `0.1.0`.
- Release targets are exactly five: `x86_64-unknown-linux-musl`, `aarch64-unknown-linux-musl`, `x86_64-apple-darwin`, `aarch64-apple-darwin`, `x86_64-pc-windows-msvc`.
- Test style: assert a hand-computed physical quantity or lock in a specific bug, with the reasoning in the doc comment.

---

## File Structure

| File | Responsibility |
|---|---|
| `crates/zc-cli/src/fit_cmd.rs` (modify) | Resolve which calibration file is *the* dataset. Gains `resolve()` + tests. |
| `data/calibration/archive/2026-08-16-pre-coverage-factor.jsonl` (create) | The retired record. Never read by `zc gate`. |
| `data/calibration/archive/README.md` (create) | Why each archived record left the active set. |
| `data/calibration/gate.jsonl` (create) | The committed cross-machine dataset backing the published number. |
| `.github/workflows/release.yml` (create) | Tag → five binaries + `.sha256` → GitHub release. |
| `docs/gate-runbook.md` (create) | Per-machine procedure for the field campaign, handable to someone else. |
| `install.sh` (create) | `curl \| sh` installer. |
| `Cargo.toml` (modify) | Fix the placeholder `repository` URL. |
| `check.sh` (modify) | Add the installer's offline self-check. |
| `README.md` (create) | Launch surface. Written last, from real `zc gate` output. |

---

## Task 1: Calibration dataset path precedence

Implements spec §A1b. `zc fit` and `zc gate` currently read one hardcoded path, `data/calibration/local.jsonl`, which is **gitignored**. The 5-machine dataset therefore has nowhere to live where CI or a reader could reproduce the gate. `.gitignore` already anticipates the successor: *"A curated cross-machine dataset would be committed under a different name."*

**Files:**
- Modify: `crates/zc-cli/src/fit_cmd.rs:8-15`
- Test: `crates/zc-cli/src/fit_cmd.rs` (new `#[cfg(test)] mod tests` at end of file)

**Interfaces:**
- Consumes: nothing.
- Produces: `fit_cmd::path() -> std::path::PathBuf` (signature unchanged — `gate_cmd.rs`, `main.rs` and `doctor.rs` all call it and must keep compiling untouched). New private `fit_cmd::resolve(dir: &std::path::Path) -> std::path::PathBuf`.

- [ ] **Step 1: Write the failing tests**

Append to `crates/zc-cli/src/fit_cmd.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::resolve;

    /// A unique scratch directory. No dev-dependencies: the workspace's
    /// dependency policy applies to tests too, so no `tempfile` crate.
    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("zc-fit-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A fresh clone ships no curated dataset, so `zc fit` must keep reading
    /// the local file rather than pointing at a path that does not exist.
    /// Without this, every new user's first `zc fit` would report no data.
    #[test]
    fn falls_back_to_local_when_no_curated_dataset() {
        let d = scratch("fallback");
        assert_eq!(resolve(&d), d.join("local.jsonl"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Once the cross-machine dataset is committed it *is* the dataset: it is
    /// what the published accuracy number is computed from, and one machine's
    /// local runs must not silently shadow it.
    #[test]
    fn curated_dataset_wins_when_present() {
        let d = scratch("curated");
        std::fs::write(d.join("gate.jsonl"), "").unwrap();
        std::fs::write(d.join("local.jsonl"), "").unwrap();
        assert_eq!(resolve(&d), d.join("gate.jsonl"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A directory entry named `gate.jsonl` that is not a file must not be
    /// selected — `is_file()` rather than `exists()` is load-bearing.
    #[test]
    fn a_directory_named_like_the_dataset_is_not_the_dataset() {
        let d = scratch("dir");
        std::fs::create_dir_all(d.join("gate.jsonl")).unwrap();
        assert_eq!(resolve(&d), d.join("local.jsonl"));
        let _ = std::fs::remove_dir_all(&d);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p zc-cli 2>&1 | tail -20`
Expected: FAIL — `cannot find function 'resolve' in this scope`.

- [ ] **Step 3: Write the implementation**

Replace lines 8–15 of `crates/zc-cli/src/fit_cmd.rs` (the `DEFAULT_PATH` const and `path()` fn) with:

```rust
const DEFAULT_DIR: &str = "data/calibration";

/// Overridable so tests and validation runs cannot contaminate a real dataset.
pub fn path() -> std::path::PathBuf {
    std::env::var("ZC_CALIBRATION")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| resolve(std::path::Path::new(DEFAULT_DIR)))
}

/// Which file in a calibration directory is the dataset.
///
/// `gate.jsonl` is the curated cross-machine set committed to the repo — it is
/// what the published accuracy figure is computed from, so CI and any reader
/// can recompute that number from a clean checkout. `local.jsonl` is gitignored
/// and holds whatever this machine's `zc verify` produced. Preferring the
/// committed file when it exists is what makes the claim reproducible; falling
/// back keeps a fresh clone with no dataset behaving exactly as before.
///
/// Merging a user's local runs *into* the shipped dataset is `zc share`'s job,
/// not this function's.
fn resolve(dir: &std::path::Path) -> std::path::PathBuf {
    let curated = dir.join("gate.jsonl");
    if curated.is_file() {
        curated
    } else {
        dir.join("local.jsonl")
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p zc-cli 2>&1 | tail -10`
Expected: PASS, 3 tests.

- [ ] **Step 5: Full check**

Run: `./check.sh`
Expected: `OK`. Note clippy runs with `-D warnings`; `resolve` is called by `path()` so there is no dead-code warning.

- [ ] **Step 6: Commit**

```bash
git add crates/zc-cli/src/fit_cmd.rs
git commit -m "feat: prefer a committed cross-machine calibration dataset

zc fit and zc gate read one hardcoded path, and that path is gitignored,
so the dataset backing the published accuracy number had nowhere to live
where CI or a reader could recompute it. Prefer data/calibration/gate.jsonl
when present, fall back to local.jsonl otherwise.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Retire the stale record and measure this machine

Implements spec §A1. `gate.rs:81` computes `slot.1 |= r.virt.as_deref() != Some("none")`, so one record with an absent `virt` marks that machine as virtualized *permanently*. The behaviour is correct and test-locked (`unknown_virt_is_not_assumed_bare_metal`), but the sole existing record has no `virt` field, so this Mac can never count toward `MIN_BARE_METAL = 2`.

The record is retired for two stated reasons — its `virt` is genuinely unknown, and it grades a superseded prediction model (pre-1.645σ coverage factor, pre-0.40 prior floor, pre-f16-KV-default). It is archived, not deleted.

**Files:**
- Create: `data/calibration/archive/2026-08-16-pre-coverage-factor.jsonl`
- Create: `data/calibration/archive/README.md`
- Create: `data/calibration/gate.jsonl`
- Delete: the single line in `data/calibration/local.jsonl`

**Interfaces:**
- Consumes: `fit_cmd::resolve()` from Task 1 — `gate.jsonl` must exist for `zc gate` to read it.
- Produces: `data/calibration/gate.jsonl`, the file Task 5's field campaign appends to.

- [ ] **Step 1: Archive the record**

```bash
mkdir -p data/calibration/archive
mv data/calibration/local.jsonl data/calibration/archive/2026-08-16-pre-coverage-factor.jsonl
```

- [ ] **Step 2: Write the archive README**

Create `data/calibration/archive/README.md`:

```markdown
# Archived calibration records

Records here are **history, not evidence**. `zc gate` and `zc fit` read a single
file (`gate.jsonl`, else `local.jsonl`); nothing in this directory is ever read.

A record is archived only for a stated reason, and "it made the number worse" is
never one of them.

## 2026-08-16-pre-coverage-factor.jsonl

One run: `qwen3:4b` Q4_K_M on Apple Silicon, -28.3% error, outside its published
range.

Retired for two reasons:

1. **Unknown provenance.** It predates the `virt` field. `zc gate` correctly
   refuses to assume an unlabelled record came from bare metal, so its presence
   marked that machine virtualized forever and barred it from `MIN_BARE_METAL`.
2. **Superseded grader.** Its `error_pct` grades a prediction made before the
   1.645σ coverage factor, the 0.40 prior floor, and the f16 KV default. The
   gate reads `error_pct` from the record rather than recomputing it — that is
   what keeps errors genuinely out-of-sample — so a record cannot be re-graded
   in place. It can only be retired and re-measured.

The machine it came from is re-measured in `gate.jsonl` under the current model.
```

- [ ] **Step 3: Start a runtime and confirm a model is available**

Ollama is installed at `/opt/homebrew/bin/ollama` but was not listening.

```bash
ollama serve >/dev/null 2>&1 &
sleep 3
ollama list
```

If `qwen3:1.7b` is absent, pull it — it is the campaign's common anchor across all five machines:

```bash
ollama pull qwen3:1.7b
```

- [ ] **Step 4: Measure this machine under the current model**

```bash
cargo run --release --bin zc -- verify qwen3:1.7b
```

Expected: exit 0, and one line appended to `data/calibration/local.jsonl` containing `"virt":"none"`.

- [ ] **Step 5: Verify the record carries bare-metal provenance**

```bash
grep -c '"virt":"none"' data/calibration/local.jsonl
```
Expected: `1`. If it prints `0`, stop — `env.rs` is misdetecting this Mac as virtualized, which would block `MIN_BARE_METAL` on every machine and is a bug to fix before continuing.

- [ ] **Step 6: Seed the committed dataset**

```bash
cp data/calibration/local.jsonl data/calibration/gate.jsonl
cargo run --release --bin zc -- gate
```

Expected output shows `1 runs on 1 machine(s)`, the machine's `kind` column reading `bare`, and the bare-metal blocker now reading `1 of 2`.

- [ ] **Step 7: Commit**

```bash
git add data/calibration/archive data/calibration/gate.jsonl
git commit -m "data: retire the pre-coverage-factor record, seed gate.jsonl

The one record on file predates the virt field, so zc gate correctly
refused to count its machine as bare metal, and it grades a prediction
from a superseded model. Archived with both reasons stated, and the
machine re-measured under the current model.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Release pipeline

Implements spec §A2. This is what makes the field campaign possible without solving local cross-compilation: GitHub's runners are natively Windows, Linux and macOS.

Uses the `gh` CLI, which is preinstalled on every GitHub-hosted runner, rather than a third-party release action — consistent with the project's supply-chain posture.

**Files:**
- Create: `.github/workflows/release.yml`
- Modify: `Cargo.toml:9` (`repository` is the placeholder `https://github.com/zerocloud/zerocloud`; the real remote is `https://github.com/DEEPESH-845/ZeroCloud`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: release assets named exactly `zc-<target>` and `zc-<target>.sha256` (with `.exe` before `.sha256` on Windows: `zc-x86_64-pc-windows-msvc.exe`, `zc-x86_64-pc-windows-msvc.exe.sha256`). Task 6's `install.sh` constructs URLs from this naming and must match it character for character.

- [ ] **Step 1: Fix the repository URL**

In `Cargo.toml`, change:

```toml
repository = "https://github.com/zerocloud/zerocloud"
```

to:

```toml
repository = "https://github.com/DEEPESH-845/ZeroCloud"
```

- [ ] **Step 2: Write the workflow**

Create `.github/workflows/release.yml`:

```yaml
# Tag -> five binaries + checksums on a GitHub release.
#
# Deliberately triggered by `push: tags` and tagged by hand. release-please
# tags using GITHUB_TOKEN, and tags pushed with that token do not trigger
# downstream workflows -- the documented way to ship a release with no assets
# attached. Hand-tagging avoids the whole class of problem until there is a
# reason to automate versioning.
name: release

on:
  push:
    tags: ["v*"]

permissions:
  contents: write

jobs:
  # The release has to exist before any matrix job can upload to it, and five
  # parallel jobs racing to create it would collide.
  create:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Create the release if it does not exist
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          if gh release view "${{ github.ref_name }}" >/dev/null 2>&1; then
            echo "release already exists"
            exit 0
          fi
          # Any tag carrying a hyphen (v0.1.0-rc1) is a prerelease.
          case "${{ github.ref_name }}" in
            *-*) pre=--prerelease ;;
            *)   pre= ;;
          esac
          gh release create "${{ github.ref_name }}" \
            --title "${{ github.ref_name }}" \
            --generate-notes --verify-tag $pre

  build:
    needs: create
    strategy:
      fail-fast: false
      matrix:
        include:
          - target: x86_64-unknown-linux-musl
            os: ubuntu-latest
          - target: aarch64-unknown-linux-musl
            os: ubuntu-latest
          - target: x86_64-apple-darwin
            os: macos-latest
          - target: aarch64-apple-darwin
            os: macos-latest
          - target: x86_64-pc-windows-msvc
            os: windows-latest
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - name: Add target
        run: rustup target add ${{ matrix.target }}

      # x86_64 musl links with the packaged musl-gcc. The runner image has no
      # packaged aarch64 cross-musl-gcc, so that target links with rust-lld
      # instead -- which works here only because nothing in this workspace
      # compiles C: `libc` is FFI declarations, not a C build.
      - name: musl linker
        if: matrix.os == 'ubuntu-latest'
        run: sudo apt-get update && sudo apt-get install -y musl-tools

      - name: Build
        env:
          CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER: rust-lld
        run: cargo build --release --target ${{ matrix.target }} --bin zc

      # `strip = true` is already set in [profile.release], so the binary
      # arrives stripped and no separate step is needed.
      - name: Package
        shell: bash
        run: |
          set -eu
          if [ "${{ runner.os }}" = "Windows" ]; then ext=".exe"; else ext=""; fi
          out="zc-${{ matrix.target }}${ext}"
          cp "target/${{ matrix.target }}/release/zc${ext}" "$out"
          if command -v sha256sum >/dev/null 2>&1; then
            sha256sum "$out" > "$out.sha256"
          else
            shasum -a 256 "$out" > "$out.sha256"
          fi
          cat "$out.sha256"

      # Smoke test: a binary that cannot emit its own report is not shippable,
      # and this is the only place the Linux and Windows probe paths run at all.
      # aarch64-linux cannot run on an x86_64 runner, so it is skipped here and
      # covered by the field campaign instead.
      - name: Smoke test
        if: matrix.target != 'aarch64-unknown-linux-musl'
        shell: bash
        run: |
          set -eu
          if [ "${{ runner.os }}" = "Windows" ]; then ext=".exe"; else ext=""; fi
          ./"zc-${{ matrix.target }}${ext}" check --json --top 3 | python3 -m json.tool > /dev/null
          echo "json parses"

      - name: Upload
        shell: bash
        env:
          GH_TOKEN: ${{ github.token }}
        run: gh release upload "${{ github.ref_name }}" zc-${{ matrix.target }}* --clobber
```

- [ ] **Step 3: Validate the workflow syntax locally**

```bash
python3 -c "import sys,yaml; yaml.safe_load(open('.github/workflows/release.yml')); print('yaml ok')" \
  || echo "install pyyaml or skip: GitHub will report syntax errors on push"
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/release.yml Cargo.toml
git commit -m "ci: build and publish five release targets on tag

GitHub's runners are natively Windows, Linux and macOS, so this replaces
local cross-compilation rather than duplicating it -- and every calibration
record from the field campaign then comes from the exact artifact users
download. Uses the preinstalled gh CLI rather than a third-party action.

Also corrects the placeholder repository URL.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Push and cut the first prerelease**

```bash
git push origin main
git tag v0.1.0-rc1
git push origin v0.1.0-rc1
gh run watch
```

- [ ] **Step 6: Verify all five assets exist**

```bash
gh release view v0.1.0-rc1 --json assets --jq '.assets[].name' | sort
```

Expected exactly ten names — five binaries and five `.sha256` files:

```
zc-aarch64-apple-darwin
zc-aarch64-apple-darwin.sha256
zc-aarch64-unknown-linux-musl
zc-aarch64-unknown-linux-musl.sha256
zc-x86_64-apple-darwin
zc-x86_64-apple-darwin.sha256
zc-x86_64-pc-windows-msvc.exe
zc-x86_64-pc-windows-msvc.exe.sha256
zc-x86_64-unknown-linux-musl
zc-x86_64-unknown-linux-musl.sha256
```

If a job failed, fix and re-tag with `-rc2` rather than force-moving `-rc1` — a moved tag makes it impossible to say which artifact produced a given calibration record.

---

## Task 4: `install.sh`

Implements spec §B2. Depends on Task 3's asset naming.

**Files:**
- Create: `install.sh`
- Modify: `check.sh` (append the installer self-check)

**Interfaces:**
- Consumes: asset names `zc-<target>[.exe]` and `zc-<target>[.exe].sha256` from Task 3.
- Produces: `ZC_DRY_RUN=1` (resolve and print, download nothing) and `ZC_VERSION=<tag>` (skip the network tag lookup) — both used by the self-check and by bug reports.

- [ ] **Step 1: Write the installer**

Create `install.sh`:

```sh
#!/bin/sh
# ZeroCloud installer.
#
#   curl -fsSL https://raw.githubusercontent.com/DEEPESH-845/ZeroCloud/main/install.sh | sh
#
# ZC_VERSION=v0.1.0  install a specific tag instead of the latest
# ZC_INSTALL_DIR=... override the first install directory tried
# ZC_DRY_RUN=1       resolve and print what would happen, download nothing
set -eu

REPO="DEEPESH-845/ZeroCloud"
BIN="zc"

die() { echo "install: $*" >&2; exit 1; }

# Resolve the latest tag by following the /releases/latest redirect rather than
# asking the API. The unauthenticated API allows 60 requests/hour per IP, and a
# shared NAT -- a school, an office, a country behind CGNAT -- burns that on
# other people's traffic. The redirect has no such limit.
latest_tag() {
    curl -fsSLI -o /dev/null -w '%{url_effective}' \
        "https://github.com/$REPO/releases/latest" | sed 's#.*/tag/##'
}

# uname -> release target triple. Fails loudly rather than guessing: installing
# the wrong architecture produces a confusing exec-format error much later.
target() {
    os=$(uname -s)
    arch=$(uname -m)
    case "$os" in
        Linux)
            case "$arch" in
                x86_64|amd64)  echo x86_64-unknown-linux-musl ;;
                aarch64|arm64) echo aarch64-unknown-linux-musl ;;
                *) die "unsupported architecture: $arch" ;;
            esac ;;
        Darwin)
            case "$arch" in
                x86_64) echo x86_64-apple-darwin ;;
                arm64)  echo aarch64-apple-darwin ;;
                *) die "unsupported architecture: $arch" ;;
            esac ;;
        *)
            die "unsupported OS: $os -- Windows users download the .exe from
    https://github.com/$REPO/releases/latest" ;;
    esac
}

# A checksum that cannot be computed is not a checksum that passes. This is a
# trust boundary: refuse rather than install an unverified binary.
verify() {
    if command -v sha256sum >/dev/null 2>&1; then
        have=$(sha256sum "$1" | cut -d' ' -f1)
    elif command -v shasum >/dev/null 2>&1; then
        have=$(shasum -a 256 "$1" | cut -d' ' -f1)
    else
        die "no sha256sum or shasum available; refusing to install unverified"
    fi
    want=$(cut -d' ' -f1 < "$2")
    [ -n "$want" ] || die "empty checksum file"
    [ "$have" = "$want" ] || die "checksum mismatch (got $have, want $want)"
}

# /usr/local/bin if writable, sudo if we can prompt, else ~/.local/bin.
install_to() {
    src=$1
    first=${ZC_INSTALL_DIR:-/usr/local/bin}
    for dir in "$first" "$HOME/.local/bin"; do
        mkdir -p "$dir" 2>/dev/null || true
        if [ -w "$dir" ]; then
            mv "$src" "$dir/$BIN" && chmod 755 "$dir/$BIN" && { echo "$dir/$BIN"; return; }
        fi
        # Prompt on the terminal, not stdin: under `curl | sh` stdin is the
        # script itself, so a password read from it would consume the script.
        if [ -r /dev/tty ] && command -v sudo >/dev/null 2>&1; then
            if sudo -v < /dev/tty; then
                sudo mv "$src" "$dir/$BIN" && sudo chmod 755 "$dir/$BIN" \
                    && { echo "$dir/$BIN"; return; }
            fi
        fi
    done
    die "no writable install directory (tried $first and $HOME/.local/bin)"
}

tgt=$(target)
tag=${ZC_VERSION:-$(latest_tag)}
[ -n "$tag" ] || die "could not resolve the latest release tag"
case "$tgt" in *windows*) ext=".exe" ;; *) ext="" ;; esac
url="https://github.com/$REPO/releases/download/$tag/$BIN-$tgt$ext"

if [ -n "${ZC_DRY_RUN:-}" ]; then
    echo "target  $tgt"
    echo "tag     $tag"
    echo "url     $url"
    exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL "$url" -o "$tmp/$BIN" || die "no asset for $tgt in $tag"
curl -fsSL "$url.sha256" -o "$tmp/$BIN.sha256" || die "no checksum for $tgt in $tag"
verify "$tmp/$BIN" "$tmp/$BIN.sha256"
chmod +x "$tmp/$BIN"

where=$(install_to "$tmp/$BIN")
dir=$(dirname "$where")
echo "installed $tag -> $where"
case ":$PATH:" in
    *":$dir:"*) ;;
    *) echo "note: $dir is not on PATH" ;;
esac
echo "run: $BIN check"
```

- [ ] **Step 2: Make it executable and syntax-check it**

```bash
chmod +x install.sh
sh -n install.sh && echo "syntax ok"
```

- [ ] **Step 3: Run the offline self-check**

`ZC_VERSION` skips the network lookup, so this exercises `target()` and URL
construction with no network and no release required:

```bash
ZC_VERSION=v0.0.0 ZC_DRY_RUN=1 sh install.sh
```

Expected on this Apple Silicon Mac:

```
target  aarch64-apple-darwin
tag     v0.0.0
url     https://github.com/DEEPESH-845/ZeroCloud/releases/download/v0.0.0/zc-aarch64-apple-darwin
```

- [ ] **Step 4: Wire the self-check into `check.sh`**

Append to `check.sh`, before the final `echo "OK"`:

```bash
echo "== installer =="
sh -n install.sh
# Offline: ZC_VERSION skips the network tag lookup, so this exercises target
# detection and URL construction without needing a published release.
ZC_VERSION=v0.0.0 ZC_DRY_RUN=1 sh install.sh | grep -q "zc-$(
  case "$(uname -s)/$(uname -m)" in
    Darwin/arm64)  echo aarch64-apple-darwin ;;
    Darwin/x86_64) echo x86_64-apple-darwin ;;
    Linux/x86_64)  echo x86_64-unknown-linux-musl ;;
    Linux/aarch64) echo aarch64-unknown-linux-musl ;;
    *) echo UNSUPPORTED ;;
  esac)" || { echo "installer resolved the wrong target"; exit 1; }
```

- [ ] **Step 5: Run the full check**

Run: `./check.sh`
Expected: ends with `OK`, and an `== installer ==` section appears.

- [ ] **Step 6: End-to-end against the real prerelease**

Requires Task 3 to have published `v0.1.0-rc1`.

```bash
ZC_INSTALL_DIR=/tmp/zc-install-test sh install.sh
/tmp/zc-install-test/zc check --top 3
rm -rf /tmp/zc-install-test
```

Expected: a checksum-verified download, then a real prediction table.

- [ ] **Step 7: Commit**

```bash
git add install.sh check.sh
git commit -m "feat: add curl|sh installer

Resolves the latest tag by redirect rather than the API, which is rate
limited to 60 req/h per IP and shared behind any NAT. Refuses to install
when no sha256 tool is available rather than skipping verification, and
reads sudo from /dev/tty because stdin is the script under a pipe.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Gate runbook and the field campaign

Implements spec §A3 and §A4. Steps 1–2 are authoring; steps 3–7 are physical work on five machines and cannot be automated.

**Files:**
- Create: `docs/gate-runbook.md`
- Modify: `data/calibration/gate.jsonl` (appended per machine)
- Create: `docs/doctor-bundles/` (one markdown file per machine)

**Interfaces:**
- Consumes: release assets from Task 3, `gate.jsonl` from Task 2, `install.sh` from Task 4.
- Produces: a `gate.jsonl` on which `cargo run --bin zc -- gate` exits 0.

- [ ] **Step 1: Write the runbook**

Create `docs/gate-runbook.md`:

```markdown
# Phase 0 gate runbook

The gate needs **≥5 distinct machines**, **≥2 on bare metal**, and a **median of
per-machine medians below 25%**. It reads `error_pct` from each record rather
than recomputing it, so every error is genuinely out-of-sample: the prediction
was made before that run existed.

Target spread, per `PLAN.md` Phase 0 — the point is coverage of failure modes,
not five of the same laptop:

| Slot | Machine | Why it is on the list |
|---|---|---|
| 1 | Apple Silicon Mac | unified memory, Metal backend |
| 2 | 8 GB Windows laptop | the primary target market; also the only test of the WMI + registry GPU path |
| 3 | Old Intel Mac | discrete or integrated Intel, non-unified memory |
| 4 | Linux desktop | the sysfs/lspci GPU path and `GetLogicalProcessorInformationEx`'s counterpart |
| 5 | Anything DRAM-less or single-channel | the failure mode a lookup table cannot see |

## Per machine

1. **Install.** macOS/Linux:

       curl -fsSL https://raw.githubusercontent.com/DEEPESH-845/ZeroCloud/main/install.sh | sh

   Windows: download `zc-x86_64-pc-windows-msvc.exe` from the releases page and
   verify it against the `.sha256` beside it.

2. **Doctor first, before installing anything else.**

       zc doctor > doctor-<label>.md

   Do this before the runtime is installed, so the probe sees the machine as a
   new user's would. These bundles are the campaign's second deliverable:
   `VERIFICATION.md` marks the Linux and Windows probe paths as never-executed,
   `cpu.rs:291`'s `GetLogicalProcessorInformationEx` is flagged `UNVALIDATED`,
   and `FILE_FLAG_NO_BUFFERING` may serialise concurrent reads. This is the only
   planned chance to exercise them on hardware someone can inspect.

   The bundle carries no hostname, username, serial, MAC or IP, and rewrites
   paths to `~`.

3. **Install one calibration-grade runtime** — Ollama, llama.cpp
   (`llama-server`) or LM Studio. vLLM, MLX and Docker Model Runner are detected
   but refused: their APIs report no prefill/decode split, so a rate measured
   through them would include HTTP and scheduling time.

4. **Pull the anchor model.**

       ollama pull qwen3:1.7b

   `qwen3:1.7b` (~1.4 GB) is the common anchor across all five machines, so the
   fit buckets have one model measured everywhere. Then measure whatever else is
   already installed locally — more models per machine is free evidence.

5. **Measure.**

       zc verify qwen3:1.7b

   Repeat per model. Each run appends one line to `data/calibration/local.jsonl`
   next to the binary's working directory.

6. **Carry back** that `local.jsonl` and the doctor bundle.

## Merging

Append each machine's lines to the committed dataset and drop the doctor bundle
in place:

    cat /path/from/machine/local.jsonl >> data/calibration/gate.jsonl
    cp /path/from/machine/doctor-<label>.md docs/doctor-bundles/

Per-machine provenance is already carried by the `hw` fingerprint, which is what
`zc gate` groups on, so no per-machine file split is needed.

## Closing

    zc fit    # what the dataset now says, and how much evidence backs it
    zc gate   # exits non-zero until it passes

`zc fit` moves coefficients from records. **Never edit a coefficient by hand**,
including when the gate is red and there is a hypothesis about which direction
it should move. That rule is what makes the published number mean anything.

## Reading the result

Two numbers matter, and the second is the one that can hurt:

- **median error < 25%** — the stated gate.
- **`within_range`** — the published promise is a range, not a point. The only
  measurement previously on record fell outside its range. The 1.645σ coverage
  factor and the 0.40 prior floor were written to fix exactly that and have
  never been tested against a real measurement.

If the median passes but `within_range` stays low, the midpoint is right and the
published *width* is wrong. That is a coverage-factor and confidence-tier
problem, and it is fixed by widening honestly — never by narrowing to fit.

**Stop-loss**, per `PLAN.md`: if neither number converges across five real
machines, the product does not work, and finding that out in week two is what
the gate was designed to do.
```

- [ ] **Step 2: Commit the runbook**

```bash
mkdir -p docs/doctor-bundles
git add docs/gate-runbook.md
git commit -m "docs: add the Phase 0 gate runbook

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: Harvest the three CI machines already being measured**

`.github/workflows/calibrate.yml` already runs `zc verify` on ubuntu-latest,
macos-latest and windows-latest, merges the three records, and uploads them as
the `calibration-merged` artifact — it simply never commits them. Those are three
distinct machines the gate can count. They are VMs, so they can never satisfy
`MIN_BARE_METAL`, which is precisely why that floor exists.

```bash
gh workflow run calibrate.yml
gh run watch
gh run download -n calibration-merged -D /tmp/ci-records
cat /tmp/ci-records/all.jsonl >> data/calibration/gate.jsonl
```

This reduces the physical campaign's *minimum* to one more bare-metal machine.
Run the full spread anyway where hardware is available: a gate that squeaks green
on three cloud VMs and two laptops is weaker evidence than five real machines,
and the extra records cost only time.

- [ ] **Step 4: Run the campaign on the remaining physical machines**

Follow `docs/gate-runbook.md` on each. Physical work: one install, one doctor
bundle, one runtime, one ~1.4 GB model pull, and one or more `zc verify` runs per
machine. **At least one more bare-metal machine is mandatory** — `MIN_BARE_METAL`
is 2 and this Mac is currently the only one.

- [ ] **Step 5: Merge the records**

```bash
cat /path/to/machine-N/local.jsonl >> data/calibration/gate.jsonl
cp /path/to/machine-N/doctor-*.md docs/doctor-bundles/
```

- [ ] **Step 6: Refit and read the gate**

```bash
cargo run --release --bin zc -- fit
cargo run --release --bin zc -- gate; echo "exit: $?"
```

Expected: `exit: 0`, ≥5 machines, ≥2 bare-metal, median below 25%.

- [ ] **Step 7: If `within_range` is low, stop and treat it as physics**

Do not proceed to the README. Widening the published range is a change to the
coverage factor or the confidence tiers in `crates/zc-model/src/fit.rs`, it needs
its own hand-computed test in the house style, and it is out of this plan's
scope — bring the numbers back for a design decision.

- [ ] **Step 8: Commit the dataset**

```bash
git add data/calibration/gate.jsonl docs/doctor-bundles
git commit -m "data: Phase 0 gate dataset -- N machines, median X.X% error

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: README

Implements spec §B1. **Blocked on Task 5 exiting 0** — the point is that the
first number is real.

Part 1 of `~/.claude/plans/i-am-providing-you-robust-donut.md` is the first draft
of the comparison material and is already written and evidenced; lift from it
rather than rewriting.

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: real output from `zc gate`, `zc check` and `zc fit`.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Capture the real output to paste**

```bash
cargo run --release --bin zc -- gate  | tee /tmp/zc-gate.txt
cargo run --release --bin zc -- check --top 8 | tee /tmp/zc-check.txt
```

- [ ] **Step 2: Write the README with this structure**

Every number below is pasted from step 1's captures. There are no placeholders —
if a number is not yet measured, the section does not ship.

1. **One-line pitch** — what can this machine run, and how fast.
2. **The accuracy number, first.** Machines, median out-of-sample error, and the
   date measured. Say plainly that `zc gate` computes it from
   `data/calibration/gate.jsonl` in this repo and that anyone can recompute it.
3. **Install** — the `curl | sh` line, plus the Windows releases-page link.
4. **Real `zc check` output above the fold**, pasted from `/tmp/zc-check.txt`.
5. **How it works** — the three measurements (`zc-bench`: 256 MiB-working-set RAM
   bandwidth swept across thread counts, O_DIRECT uncached disk, 64-lane FMA),
   then the memory model, then decode as bandwidth-bound.
6. **The rule** — a number is measured, derived from measured inputs, or printed
   as `-`. Name the consequences: `ttft_s` is `null` until measured because
   deriving it from the FMA benchmark was measured wrong by 10–40×; catalog
   entries with an unrecognised attention kind are rejected rather than guessed.
7. **What it does not do** — no network by default, no telemetry, no model
   downloads, no browser hardware scan.
8. **Comparison to llmfit**, honest in both directions: llmfit is wider (more
   models, more surfaces, a TUI, a server); ZeroCloud measures the machine
   instead of looking it up, models MLA/SWA/Hybrid KV rather than applying the
   GQA formula to everything, and publishes an out-of-sample accuracy figure.
9. **Contributing** — `data/models/*.json` and calibration records are the
   contributions that matter; point at `docs/gate-runbook.md`.

- [ ] **Step 3: Verify every claim in the README resolves**

```bash
grep -o 'docs/[a-z-]*\.md' README.md | sort -u | while read -r f; do
  [ -f "$f" ] || echo "MISSING: $f"
done
```
Expected: no output.

- [ ] **Step 4: Commit and tag the real release**

```bash
git add README.md
git commit -m "docs: add README

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push origin main
git tag v0.1.0
git push origin v0.1.0
```

---

## Out of scope

Named so nobody adds them mid-plan:

- `zc share`, `zc plan`, live HF lookup, `zc serve`, `zc-tui`, web cards, §2.6's fit-score falloff. All Phase C, all post-launch.
- CSV renderer, Homebrew/winget, code signing, release-please, Docker image.
- Any hand-tuned coefficient.
