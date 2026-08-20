# Security

## Reporting a vulnerability

Use GitHub's private vulnerability reporting:
**[Report a vulnerability](https://github.com/DEEPESH-845/ZeroCloud/security/advisories/new)**

That opens a private thread with the maintainer and a private fork to fix in.
Please do not open a public issue for anything you believe is exploitable.

Expect a first response within a week. This is a small project with one
maintainer, so the honest answer about timelines is that a fix ships when it is
ready and you will be told where it stands.

## What `zc` actually does

Worth knowing before you look for a boundary to test:

- **It runs other programs.** `nvidia-smi`, `lspci` and `powershell` to find
  GPUs; `curl` for `zc check <hf-repo-id>`; `open` / `xdg-open` / `rundll32`
  to open a browser for `zc share`. All are spawned directly with arguments —
  never through a shell — so an argument cannot become a command.
- **It reads files you point it at.** The disk benchmark reads a large existing
  file on the model volume, and creates a 512 MiB scratch file only when it
  finds none. It never writes to a file it did not create.
- **It parses input it did not write.** Hugging Face API responses, and
  calibration records that arrive as community pull requests and are compiled
  into the binary. Those parsers are fuzzed in
  `crates/zc-model/tests/fuzz.rs`.
- **It opens exactly one outbound connection, and only when asked by name.**
  `zc check <hf-repo-id>` fetches that repository's public metadata and prints
  every URL before fetching it. Nothing about your machine is sent, and there
  is no telemetry of any kind.
- **`zc share` sends nothing itself.** It prints the record field by field and
  hands a prefilled URL to your browser. You see the payload before it moves.

## What is in scope

Anything that lets a crafted model file, calibration record, Hugging Face
response, filename or environment variable cause `zc` to execute code, write
outside the paths above, or leak the contents of files it was not pointed at.

A wrong prediction is a bug, not a vulnerability — report those as issues, and
`zc doctor` output helps.

## Supported versions

The latest release. This project is pre-1.0 and fixes go forward, not back.
