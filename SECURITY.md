# Security Policy

## Supported versions

Promptline is pre-1.0. Security fixes are applied to the latest `main` and the
most recent release.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report privately via GitHub's [private vulnerability reporting](https://github.com/sanskarpan/promptline/security/advisories/new)
(Security → Report a vulnerability), or email the maintainer. Include:

- a description and impact,
- steps to reproduce (a `PROMPTLINE_FAKE_SCRIPT` repro is ideal — no API key needed),
- affected version (`promptline --version`).

You can expect an acknowledgement within a few days. Once a fix is available we
will coordinate disclosure.

## Handling secrets

Promptline never stores your `OPENROUTER_API_KEY`; it is read from the
environment at runtime and sent only to OpenRouter. The SQLite call cache stores
prompts, responses, and token/cost metadata — treat `.promptline/` as
potentially sensitive and keep it out of version control (it is gitignored by
default).

## Network exposure

The Promptline control plane (`promptline serve`) is designed for **local or
trusted-network use only**. By design it reads local files whose paths are
named in request bodies (`data_path`, `dev_path`, `val_path`) so that the CLI,
TUI, and dashboard can drive optimization and gating against on-disk datasets.
It performs **no path confinement** on those values — this keeps relative-path
workflows working, but means a caller who can reach the server can ask it to
read arbitrary local files it has access to (local file inclusion).

For this reason:

- `serve` defaults to binding `127.0.0.1` (loopback only).
- Binding to a non-loopback host (e.g. `--host 0.0.0.0`) prints a loud warning.
- Do **not** expose the control plane to untrusted networks. If you need remote
  access, put it behind an authenticating reverse proxy or an SSH tunnel and
  restrict who can reach it.
