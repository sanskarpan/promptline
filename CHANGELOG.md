# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

Hardening from an adversarial bug-hunt across the whole stack (no correctness
issues were found in the core gate statistics or optimizer math; these are
edge-case and graceful-degradation fixes):

- **Eval harness** now isolates metric exceptions per-example (a raising judge
  metric no longer aborts the whole run); the same guard was added to GEPA and
  BootstrapFewShot direct metric calls.
- **GEPA** checkpoints the partial-vector repair on resume, so a run truncated
  mid-`full_eval` no longer reloads stale zero-padded score vectors.
- **ProTeGi / BootstrapFewShot** always include `best.id` in `scores`.
- **`Signature.parse_output`** no longer truncates output values that contain
  `[[` (only a well-formed `[[field]]:` marker ends a value).
- **OpenRouter adapter** maps `content: null` responses to `LLMError` instead of
  leaking a pydantic `ValidationError`.
- **PromptProgram** fails cleanly when a module receives none of its declared
  inputs (previously sent an empty prompt).
- **Calibrator** clamps binned human labels to the judge scale, fixing an
  `IndexError` / silent confusion-matrix corruption when a declared
  `label_range` didn't contain the observed labels.
- **PairwiseJudge** no longer misreads the article "a" in prose as verdict `A`.
- **Server**: `POST /gate` with missing paths returns 400 (not 500);
  `GET /judges/certificates` skips malformed cert files; malformed YAML configs
  produce a clean CLI error instead of a traceback; non-positive `--budget` is
  rejected; `serve` warns when bound to a non-loopback host (see `SECURITY.md`).
- **Dashboard**: SSE reconnects are idempotent (no more double-counted events),
  unknown `/ui/*` routes render a proper not-found page, malformed events can't
  crash the run view, and degenerate certificates render `—` instead of `NaN`.

## [0.1.0] — 2026-07-06

Initial release. The full `calibrate → optimize → gate → serve` pipeline.

### Added

- **Core library**: declarative signatures/programs with per-module trace
  recording, an `LLMClient` protocol with an OpenRouter adapter (BYO key, pooled
  connections, retries with backoff, per-call cost tracking), a SQLite call
  cache, and an eval harness with hard rollout/cost budget walls.
- **Optimizers (from scratch, one contract)**: GEPA (per-instance Pareto
  selection, reflective mutation, system-aware merge, resumable checkpoints),
  MIPRO (bootstrap demos → grounded instruction proposals → TPE search),
  BootstrapFewShot (+ random search), ProTeGi (textual gradients + CAPO-style
  racing), and OPRO.
- **Judge subsystem**: pointwise and pairwise rubric judges (CoT-before-verdict,
  anti-verbosity, position-swap debiasing, k-sampling), agreement metrics
  (Cohen's κ, Spearman, pairwise accuracy), a calibrator that certifies the judge
  against human labels, and meta-optimization of the judge prompt. The calibrated
  judge is the default optimization/gate metric, and a passing certificate is
  required to optimize.
- **Statistical deploy gate**: paired bootstrap CIs, Holm correction across
  candidates, independent validation-split confirmation, verbosity tripwires, and
  refusals for uncalibrated judges, undersized splits, and dev/val contamination.
- **Registry & serving**: SQLite-backed versioned prompt registry with lineage,
  an active pointer only the gate advances, rollback, and a FastAPI serving plane
  (`GET /prompts/{program}/active` with ETag) plus a control plane with SSE run
  streaming.
- **Interfaces**: a Typer CLI, a Textual TUI cockpit, and a React + Vite
  dashboard (Runs, Lineage, Judge, Gate, Registry) served statically by FastAPI.
- **Data**: JSONL schema + adapters, and loaders for HelpSteer2, Bitext, and
  MT-Bench human judgments.
- **Demo**: `promptline demo setup` (online and `--offline`) for the
  support-assistant walkthrough.
- **Tests**: 444 offline unit/integration tests, an offline end-to-end suite,
  Playwright dashboard e2e, and an opt-in live smoke test.

[Unreleased]: https://github.com/sanskarpan/promptline/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sanskarpan/promptline/releases/tag/v0.1.0
