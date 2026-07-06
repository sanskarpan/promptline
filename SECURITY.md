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
