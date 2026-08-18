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

# GitHub's /releases/latest ignores prereleases. With only prereleases
# published it redirects to /releases, which has no /tag/ segment, so the sed
# above passes the whole URL through -- non-empty, so an existence check does
# not catch it, and the download then 404s against a URL with an https:// in
# the middle of it. A tag has no slashes.
valid_tag() {
    case "$1" in
        ""|*/*|*" "*) return 1 ;;
        *) return 0 ;;
    esac
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
valid_tag "$tag" || die "no published release yet.

    Prereleases are not served by /releases/latest. If one is what you want,
    pick its tag from https://github.com/$REPO/releases and rerun:

        ZC_VERSION=<tag> curl -fsSL .../install.sh | sh"
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
