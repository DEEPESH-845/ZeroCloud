# Publishing a release

Everything a user downloads comes from one place: a git tag pushed to
`DEEPESH-845/ZeroCloud`. `.github/workflows/release.yml` does the rest.

## Before the first public release

These are one-time and none of them are optional.

- [ ] **The gate.** `zc gate` must exit 0. It currently exits 1 —
      `1 of 2 bare-metal machines`. Cloud runners cannot close this; it needs a
      second physical machine running `zc verify`. See `docs/gate-runbook.md`.
- [ ] **Set the GitHub repo description and topics.** They are what people see
      in search and in the sidebar. Topics: `llm`, `local-llm`, `ollama`,
      `llama-cpp`, `benchmark`, `rust`, `cli`, `hardware`.
      ```sh
      gh repo edit --description "What can this laptop actually run, and how fast? Measures your machine and predicts local LLM speed. No network." \
        --add-topic llm --add-topic local-llm --add-topic ollama \
        --add-topic llama-cpp --add-topic benchmark --add-topic rust \
        --add-topic cli --add-topic hardware --homepage ""
      ```
- [ ] **Turn on branch protection for `main`** so the CI you just added is
      actually a gate and not decoration:
      ```sh
      gh api -X PUT repos/DEEPESH-845/ZeroCloud/branches/main/protection \
        -f 'required_status_checks[strict]=true' \
        -F 'required_status_checks[contexts][]=test (ubuntu-latest)' \
        -F 'required_status_checks[contexts][]=test (macos-latest)' \
        -F 'required_status_checks[contexts][]=test (windows-latest)' \
        -F 'enforce_admins=false' \
        -F 'required_pull_request_reviews=null' \
        -F 'restrictions=null'
      ```

## Cutting a release

```sh
# 1. Everything green, from a clean tree.
git status --porcelain          # must be empty
./check.sh                      # tests, clippy, 3 cross targets, installer

# 2. The version in the tag and the version in the binary must agree.
#    Bump `version` in the workspace [workspace.package] if this is a new one.
grep '^version' Cargo.toml
cargo build --release && ./target/release/zc --version

# 3. Tag and push. A tag with a hyphen (v0.2.0-rc1) publishes as a prerelease.
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0

# 4. Watch it build five targets.
gh run watch "$(gh run list --workflow=release.yml --limit 1 --json databaseId -q '.[0].databaseId')"

# 5. Confirm all ten assets landed (five binaries, five .sha256).
gh release view v0.1.0 --json assets -q '.assets[].name'
```

Expected asset names — `install.sh` builds these exact strings, so a rename
here breaks every `curl | sh` and nothing else will tell you:

```
zc-x86_64-unknown-linux-musl        zc-aarch64-unknown-linux-musl
zc-x86_64-apple-darwin              zc-aarch64-apple-darwin
zc-x86_64-pc-windows-msvc.exe
```

## Verify the way a user experiences it

Do this on a machine that is not the one you built on, from a directory that is
not the repo. Most of the bugs this project has shipped were only visible from
outside the repo.

```sh
cd /tmp
curl -fsSL https://raw.githubusercontent.com/DEEPESH-845/ZeroCloud/main/install.sh | sh
zc --version        # must match the tag
zc check --top 5    # must NOT say "no calibration data yet"
zc check | head -3  # must not panic
zc fit              # names the shipped dataset, not a path that does not exist
zc chekc            # suggests `zc check` rather than only refusing

# Interactive. The TUI is default-on, so this is what most users meet first.
zc                  # a table opens; arrows move, enter explains a row,
                    # / filters, s sorts, a toggles quants, ? lists keys
                    # q must restore the shell AND leave the report behind
ZC_ASCII=1 zc       # renders with * o . and -\|/ instead of box drawing
zc --tui < /dev/null   # must fail with exit 2, not open anything

# Contracts a script depends on. All four must hold on the shipped binary.
zc check | grep -c $'\033'   # 0 -- no escape sequences in piped stdout
zc check --json | python3 -m json.tool > /dev/null   # valid JSON
zc check 2>&1 >/dev/null | wc -c                     # 0 -- stderr silent when piped
zc check --json | grep -c "$HOME"                    # 0 -- no account name
```

Resize the window while the table is open, and shrink it below 40x10: it must
reflow, and then say the terminal is too small rather than painting something
unreadable. `check.sh` covers the scriptable half of this on every commit; the
interactive half needs a human and a real terminal, which is why it is here.

A prerelease is not served by `/releases/latest`, so `curl | sh` will report
`no published release yet` until a tag without a hyphen exists. To test a
prerelease: `ZC_VERSION=v0.1.0-rc1 curl ... | sh`.

## Windows

`zc.exe` is unsigned, so SmartScreen shows *"Windows protected your PC"* on
first run and the user has to click *More info → Run anyway*. This measurably
kills conversion on the platform that is the primary target market.

The order that costs the least:

1. **Ship the `.exe` on the release page now**, with the SmartScreen click-path
   written into the README's Windows section. Do not pretend it will not happen.
2. **Submit to `winget`** once a stable tag exists. `winget` installs get the
   Microsoft-signed installer path and skip the warning entirely. It is a PR to
   `microsoft/winget-pkgs` with a manifest pointing at the release asset and its
   SHA256 — no cost, and it is the single highest-leverage Windows step.
3. **Buy an EV code-signing certificate (~$400/yr) only if Windows conversion
   is measurably the bottleneck** after winget. Not before.

## macOS

The downloaded binary carries a quarantine attribute, so the first run is
blocked by Gatekeeper. `install.sh` uses `curl`, which does *not* set the
quarantine flag, so users who install that way are unaffected — this only hits
people who download the asset from the browser. The README's manual path should
say:

```sh
xattr -d com.apple.quarantine ./zc
```

Notarisation requires a $99/yr Apple Developer account. Same rule as Windows:
only if it is measurably the bottleneck.

## Homebrew

Do not open a `homebrew-core` PR — it requires notability the project does not
have yet and it will be closed. A tap is a repository with one file and works
immediately:

```
DEEPESH-845/homebrew-zerocloud/Formula/zc.rb
```

```ruby
class Zc < Formula
  desc "What can this laptop actually run, and how fast?"
  homepage "https://github.com/DEEPESH-845/ZeroCloud"
  version "0.1.0"
  license "Apache-2.0"

  on_macos do
    on_arm do
      url "https://github.com/DEEPESH-845/ZeroCloud/releases/download/v0.1.0/zc-aarch64-apple-darwin"
      sha256 "..."   # from the .sha256 asset
    end
    on_intel do
      url "https://github.com/DEEPESH-845/ZeroCloud/releases/download/v0.1.0/zc-x86_64-apple-darwin"
      sha256 "..."
    end
  end

  def install
    bin.install Dir["zc-*"].first => "zc"
  end

  test do
    assert_match "zc #{version}", shell_output("#{bin}/zc --version")
  end
end
```

Then `brew install DEEPESH-845/zerocloud/zc`. Revisit `homebrew-core` after a
few hundred stars.

## crates.io — blocked, and why

`cargo install zc-cli` does **not** work today, and publishing would fail:

```
$ cargo package -p zc-model
error: failed to verify package tarball
  cannot read ../../data/models: No such file or directory
```

`crates/zc-model/build.rs` embeds `data/models/*.json` and `fit.rs` embeds
`data/calibration/gate.jsonl`, both of which live *above* the package root. A
published `.crate` contains only its own directory, so the build script finds
nothing.

The fix is a directory move, not a code change: relocate `data/` to
`crates/zc-model/data/` and update `build.rs`, `fit.rs`, `fit_cmd.rs`,
`catalog.rs`, `.gitignore`, both workflows, `CONTRIBUTING.md` and
`docs/gate-runbook.md`. Every crate would also need `description`, `repository`
and concrete `version =` on its path dependencies.

That is worth doing when Rust developers ask for `cargo install`. It is not
worth doing before launch: the binary path already covers everyone, and moving
the community's contribution surface out of the repo root right before asking
for contributions is the wrong trade. `cargo binstall zc-cli` would still need
the crate published, so it inherits the same block.

## After the tag

- [ ] Update the README's accuracy table if the dataset changed
      (`zc gate` prints all three numbers).
- [ ] Post the release to r/LocalLLaMA with the terminal output inline, not a
      screenshot — people paste their own output back, and that is the fastest
      calibration data this project can get.
