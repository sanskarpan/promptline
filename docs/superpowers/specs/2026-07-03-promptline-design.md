# Promptline — Design Spec

**Date:** 2026-07-03
**Status:** Approved design, pre-implementation

## What & Why

Promptline is an open-source, pip-installable pipeline for automatic prompt optimization:

1. **Calibrate** an LLM-as-judge against human labels (with a measurable agreement certificate).
2. **Optimize** system prompts with from-scratch implementations of state-of-the-art algorithms (GEPA flagship).
3. **Gate** deployment on statistically significant improvement over the incumbent.
4. **Serve** the active prompt from a versioned registry via an HTTP endpoint.

**Differentiation (research-validated):** existing tools are either optimizers that trust their metric blindly (DSPy, gepa, promptim, AdalFlow) or eval platforms that inspect judges but don't optimize (promptfoo, Braintrust, Weave). No open-source tool chains *calibrated judge → optimizer → statistical deploy gate*. Promptline is that chain.

**Goals:** portfolio-quality showcase, reusable OSS library, deep learning vehicle (algorithms implemented from scratch, not wrapped).

**Non-goals (YAGNI):** multi-tenant SaaS, billing/credits/markup, accounts. Users bring their own OpenRouter API key. A hosted/billed layer is a possible later sub-project, deliberately out of scope.

## Architecture

Library-core with thin shells. Everything lives in one Python package; CLI, TUI, and FastAPI server are thin layers over the same core. The web dashboard is a static React app served by FastAPI.

```
┌─────────────────────────────────────────────┐
│  Interfaces (thin, replaceable)             │
│  ┌─────────┐ ┌─────────┐ ┌───────────────┐  │
│  │   CLI   │ │   TUI   │ │ Web dashboard │  │
│  │ (Typer) │ │(Textual)│ │ (React/Vite)  │  │
│  └────┬────┘ └────┬────┘ └───────┬───────┘  │
│       │           │        FastAPI + SSE    │
│  ─────┴───────────┴──────────────┴────────  │
│              Python core library            │
│  ┌──────────┐ ┌───────┐ ┌────────────────┐  │
│  │Optimizers│ │ Judge │ │ Eval harness   │  │
│  │GEPA/MIPRO│ │+calib.│ │ + stat gate    │  │
│  └──────────┘ └───────┘ └────────────────┘  │
│  ┌──────────┐ ┌───────────────────────────┐ │
│  │OpenRouter│ │ Registry (SQLite+files)   │ │
│  │ adapter  │ │ versions, lineage, active │ │
│  └──────────┘ └───────────────────────────┘ │
└─────────────────────────────────────────────┘
```

**Data flow:** load dataset → calibrate judge (certificate) → optimizer evolves prompts using judge as metric → gate compares winner vs incumbent → registry advances active pointer → FastAPI serves active prompt.

## Core Abstractions

### 1. PromptProgram
The unit being optimized. A program has one or more **modules**, each with a named, versioned instruction string (the system prompt) and optional few-shot demos. Structured I/O declared DSPy-signature-style (named input/output fields) without the DSPy dependency. Execution records **traces** (per-module inputs/outputs) — required by GEPA (reflection input) and MIPRO (demo mining).

### 2. Dataset
Evaluation examples with a documented JSONL schema: `{conversation, reference_output?, human_label?}`. Ships with loaders for HelpSteer2, Bitext customer support, and MT-Bench human judgments, plus an adapter interface for bring-your-own conversation logs.

### 3. Judge
An LLM evaluator that is itself a `PromptProgram` (so it can be optimized by our own optimizers — the Dropbox-validated meta-optimization pattern). Design defaults from research:
- Per-criterion rubric judges (several narrow judges beat one "overall 1–10" judge)
- Chain-of-thought reasoning first, structured verdict last
- Pairwise mode runs both orderings and averages (position-bias mitigation)
- Different model family than the generator by default (self-preference bias)
- Temperature 0, or k-sample majority vote (configurable)

### 4. Calibrator
Fits and validates a judge against human labels:
- Splits gold data into judge-dev / judge-holdout
- Reports Cohen's κ, Spearman/Pearson, pairwise accuracy on the holdout
- Can meta-optimize the judge prompt against judge-dev using any registered optimizer
- Emits a **calibration certificate** (metrics + threshold check, default κ ≥ 0.6); the pipeline refuses to run optimization with an uncertified judge

### 5. Optimizer
Pluggable strategy over a shared contract: `optimize(program, dataset, metric, budget) → candidates with lineage`. Metric contract follows GEPA's richer signature: returns `(score, feedback_text)`, optionally module-targeted.

Lineup (from-scratch implementations):

| Optimizer | Role | Key mechanics |
|---|---|---|
| **GEPA** | Flagship | Per-instance Pareto frontier selection, reflective mutation from execution traces + textual feedback, system-aware merge (≤5), minibatch b=3, rollout-budget accounting |
| **MIPRO-like** | Joint instruction+demo search | Bootstrap demos (rejection sampling) → grounded instruction proposal (dataset summary, program summary, history, random tips) → TPE/Optuna over categorical space, minibatch=35, periodic full evals |
| **BootstrapFewShot(+RandomSearch)** | Cheap tier | Teacher traces passing the metric become demos; random search over demo subsets. Reused by MIPRO stage 1 |
| **ProTeGi+racing** | Textual-gradient ablation | Failure critique → counter-edit → paraphrase expansion, beam + UCB/racing minibatch eval (CAPO's trick). The "reflection without evolution" ablation vs GEPA |
| **OPRO** | Baseline | ~200-line trajectory-extrapolation baseline; documented caveat: needs a strong optimizer model |

All optimizers emit typed run events (`candidate_proposed`, `minibatch_scored`, `pareto_updated`, `merge_attempted`, `budget_tick`) consumed by TUI/dashboard via SSE.

Research note: skip PromptBreeder/EvoPrompt (GEPA dominates that niche 10–60× cheaper), TextGrad/AdalFlow (framework surface, core idea already in GEPA/ProTeGi), PromptAgent (MCTS cost without payoff), COPRO (dominated by MIPRO). SPO (reference-free pairwise, no labels needed) is a possible later addition.

### 6. Eval harness + budget
Shared by all optimizers: concurrent rollout execution, every metric call counted against a hard rollout/token/cost budget, per-call cost tracking (OpenRouter usage), SQLite LLM-call cache keyed on model+messages+params (makes reruns/resumes cheap and enables cassette-style tests).

### 7. Gate + Registry
- **Gate:** candidate vs incumbent on the same eval examples → per-example paired deltas → bootstrap CI (~10k resamples) must exclude zero; Holm correction across multiple candidates; winner confirmed on an untouched validation split before promotion. Refuses undersized eval sets (configurable minimum + power warning) and contaminated splits.
- **Registry:** SQLite + prompt files. Every prompt version stored with scores, lineage (parent pointers — GEPA's evolution tree falls out naturally), and an `active` pointer only the gate can advance. Supports rollback.
- **Judge-hacking tripwires:** gate report flags significant output-length increases (verbosity exploit) and samples N outputs for optional human spot-check.

### 8. LLMClient
Single interface for all LLM traffic. First adapter: **OpenRouter** (BYO key → all major models). Retries with exponential backoff on 429/5xx, one repair-reprompt for malformed structured output, then the rollout scores as failed (never crashes a run). Distinct model roles: task model (cheap), reflection/proposer model (strong), judge model (different family than task model).

## Interfaces

### CLI (`promptline`, Typer)
`init` (scaffold `promptline.yaml`: task, dataset, models, budget, gate thresholds) · `calibrate` · `optimize --optimizer gepa --budget 500` (streams progress, `--resume <run-id>`) · `gate <candidate-id>` · `registry list|show|activate|rollback` · `serve` · `demo setup` · `tui`.

### TUI (Textual)
Live cockpit for a running optimization: score curve, per-example Pareto grid, candidate lineage tree, live rollout log with per-call cost, budget burn-down. Attaches to live runs (event stream) or finished runs (post-hoc).

### FastAPI server
- **Control plane:** REST mirroring the CLI (start runs, list candidates, run gate, manage registry) + SSE run-event stream.
- **Serving plane:** `GET /prompts/{program}/active` with ETag — the production "deploy" target apps poll.

### Web dashboard (React + Vite, static build served by FastAPI)
Pages: **Runs** (live score curves, budget/cost meters, event feed) · **Lineage explorer** (evolution tree; click a node → full prompt, diff vs parent, per-example scores — the signature visual) · **Judge** (calibration certificate, judge-vs-human scatter/confusion matrix, disagreement browser) · **Gate report** (paired delta histogram, bootstrap CI, verdict, promote/rollback) · **Registry** (version history, active pointer).

### Design language (dashboard + TUI)
Terminal-native aesthetic in the style of opencode / Hermes: dark theme by default, monospace typography throughout, flat panels with sharp corners and 1px borders, muted accent palette (single accent color + semantic green/red for gate verdicts), dense data-first layouts, no gradients/glassmorphism/rounded-card fluff. The web dashboard should feel visually continuous with the TUI — same palette and type, richer interactivity.

## Demo Story (`examples/support-assistant/`)

1. `promptline demo setup` — downloads HelpSteer2 (judge gold set, CC-BY-4.0, per-attribute human ratings) + Bitext customer support (task data), builds splits.
2. Calibrate a helpfulness judge on HelpSteer2 → certificate with κ vs human raters, benchmarked against the MT-Bench GPT-4-judge baseline (~80% human agreement).
3. Optimize a deliberately-mediocre seed support-assistant prompt with GEPA, budget capped to a few dollars on cheap OpenRouter models.
4. Gate the winner, promote, `curl` the serving endpoint.

README leads with the before/after prompt diff and the gate report.

## Error Handling

- LLM retries/repair as above; all calls cached (SQLite) so crashed runs resume cheaply.
- **Resumable runs:** optimizer state (candidate pool, scores, budget spent) checkpoints after every accepted candidate; Ctrl-C is a clean checkpoint.
- **Budget is a hard wall**, enforced centrally in the eval harness; budget exhaustion ends the run gracefully with best-so-far.
- Gate input validation: certified judge required, minimum eval-set size, split-contamination check.

## Testing

- **Unit tests with `FakeLLMClient`** (deterministic scripted responses): Pareto selection, merge triplet rules, TPE search, gate math tested exactly offline. Statistical functions validated against known distributions (bootstrap CI coverage on synthetic data).
- **Golden-trace tests:** recorded mini-runs (from the LLM cache, cassette-style) replayed in CI per optimizer.
- **Integration smoke test** (opt-in, needs key): 5 examples / 10 rollouts against a cheap model, asserts pipeline completes with a gate report.
- **Property tests:** registry activate/rollback invariants, dataset adapter schema round-trips.
- API via FastAPI TestClient; dashboard component tests + one Playwright happy-path.

## Repo Layout

```
promptline/
  core/          # PromptProgram, signatures, traces, LLMClient, OpenRouter adapter, cache
  data/          # Dataset, JSONL schema, HelpSteer2/Bitext/MT-Bench loaders
  judge/         # Judge, Calibrator, certificates, agreement metrics
  optimizers/    # base contract, gepa/, mipro/, bootstrap/, protegi/, opro/
  eval/          # harness, budget, metrics, stats (bootstrap, Holm, power)
  registry/      # SQLite registry, gate
  server/        # FastAPI app (control + serving), SSE events
  tui/           # Textual app
  cli/           # Typer commands
web/             # React dashboard (built → served by FastAPI)
examples/support-assistant/
tests/
docs/
```

Stack: Python 3.11+, `uv`, Ruff + pyright, MIT license.

## Build Phases (each independently shippable)

1. **Core + BootstrapFewShot + OPRO** — abstractions, OpenRouter client, eval harness, cache, CLI. Two simple optimizers prove the contract.
2. **Judge + Calibrator + datasets** — loaders, calibration certificates.
3. **GEPA + ProTeGi** — flagship optimizers with lineage/events.
4. **Gate + Registry + serving endpoint.**
5. **MIPRO-like** (reuses phase-1 bootstrap + Optuna).
6. **TUI, then dashboard** (terminal-native design language).
7. **Demo polish + docs + benchmark writeup** (GEPA vs ProTeGi vs MIPRO on the demo task).

## Key References

- GEPA: arXiv 2507.19457 · github.com/gepa-ai/gepa
- MIPROv2: arXiv 2406.11695 · dspy.ai
- ProTeGi/APO: EMNLP 2023 · CAPO racing: arXiv 2504.16005
- OPRO: arXiv 2309.03409 (caveat: arXiv 2405.10276)
- Judge biases: Zheng et al. arXiv 2306.05685 · EvalGen arXiv 2404.12272 · Judge's Verdict arXiv 2510.09738
- Judge meta-optimization precedent: Dropbox Dash relevance judge (dropbox.tech)
- Datasets: HelpSteer2 (arXiv 2406.08673, CC-BY-4.0) · Bitext (HF, CDLA-Sharing-1.0) · MT-Bench human judgments (CC-BY-4.0)
