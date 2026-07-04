# PROMPTLINE

**Calibrate → Optimize → Gate → Serve.** Promptline is an open-source, pip-installable pipeline for automatic prompt optimization: it calibrates an LLM-as-judge against human labels (with a measurable agreement certificate), evolves your system prompt with from-scratch implementations of state-of-the-art optimizers (GEPA flagship), refuses to deploy anything that isn't a statistically significant improvement over the incumbent, and serves the active prompt from a versioned registry over HTTP. Existing tools are either optimizers that trust their metric blindly (DSPy, gepa, promptim, AdalFlow) or eval platforms that inspect judges but don't optimize (promptfoo, Braintrust, Weave) — no other open-source tool chains *calibrated judge → optimizer → statistical deploy gate*. Promptline is that chain.

## Before / after

The demo starts from a deliberately mediocre seed and lets GEPA earn its keep:

```diff
- You are a support agent. Answer the question.
+ You are a senior customer-support agent. For every customer message:
+ 1. Acknowledge the specific problem in one empathetic sentence — never a
+    generic apology.
+ 2. Give the concrete resolution or exact next steps (menu paths, timelines,
+    fees), pulling details from the conversation rather than restating policy.
+ 3. If information is missing (order number, email), ask for exactly what you
+    need and say what you'll do once you have it.
+ 4. Close by offering the relevant follow-up action (refund, replacement,
+    escalation) — do not deflect to "contact support"; you are support.
+ Keep answers under 120 words. Do not pad, upsell, or ask for reviews.
```

…and the gate only promotes it if the paired bootstrap CI on a held-out split excludes zero.

## Quickstart

```bash
uv pip install -e ".[data]"
export OPENROUTER_API_KEY=sk-or-...   # bring your own key → all major models

promptline demo setup                 # datasets + config (see examples/support-assistant/)
cd examples/support-assistant/workspace

promptline calibrate --gold gold.jsonl --label-min 0 --label-max 4
promptline optimize --optimizer gepa --data train.jsonl
promptline gate --candidate <best_id> --dev dev.jsonl --val val.jsonl
promptline serve                      # GET /prompts/support/active
```

No key? `promptline demo setup --offline` plus the `PROMPTLINE_FAKE_SCRIPT` fake client rehearses the whole pipeline deterministically — see [examples/support-assistant/README.md](examples/support-assistant/README.md).

## Architecture

Library-core with thin shells: CLI, TUI, and FastAPI server are thin layers over one Python package; the web dashboard is a static React app served by FastAPI.

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

## The pipeline

```
 gold labels          train.jsonl              dev + val splits
      │                    │                          │
      ▼                    ▼                          ▼
 ┌───────────┐      ┌────────────┐      ┌──────────────────────┐
 │ calibrate │─────▶│  optimize  │─────▶│         gate         │
 │ judge vs  │ cert │ GEPA/MIPRO │ best │ paired bootstrap CI  │
 │ humans, κ │      │ /ProTeGi/… │ cand │ + Holm + val confirm │
 └───────────┘      └────────────┘      └──────────┬───────────┘
                                          promote  │  reject
                                                   ▼
                                        ┌──────────────────────┐
                                        │  registry (SQLite)   │──▶ serve
                                        │  versions + lineage  │    GET /prompts/
                                        │  + active pointer    │    {program}/active
                                        └──────────────────────┘    (ETag)
```

Every stage refuses bad inputs: uncalibrated judges (when a certificate is required), undersized eval sets, contaminated dev/val splits.

## Optimizers

All implemented from scratch against one contract (`optimize(program, seed, trainset, metric, budget, harness, emit)`), all budget-metered, all emitting typed run events.

| Optimizer | One-liner | Paper |
|---|---|---|
| **GEPA** (flagship) | Per-instance Pareto frontier + reflective mutation from execution traces, system-aware merges, strict minibatch acceptance | [arXiv 2507.19457](https://arxiv.org/abs/2507.19457) |
| **MIPRO** | Bootstrapped demo sets × grounded instruction proposals, searched jointly with TPE (Optuna) and periodic full evals | [arXiv 2406.11695](https://arxiv.org/abs/2406.11695) |
| **BootstrapFewShot (+RS)** | Teacher traces that pass the metric become few-shot demos; random search over demo subsets | DSPy lineage, same paper as MIPRO |
| **ProTeGi** | Textual gradients: critique failures → counter-edit → paraphrase, with CAPO-style successive-halving racing | [EMNLP 2023](https://arxiv.org/abs/2305.03495), racing: [arXiv 2504.16005](https://arxiv.org/abs/2504.16005) |
| **OPRO** | Trajectory extrapolation: show (instruction, score) history sorted worst→best, ask for a better one. Needs a strong proposer model | [arXiv 2309.03409](https://arxiv.org/abs/2309.03409) |

Concept docs: [core](docs/concepts/core.md) · [judge](docs/concepts/judge.md) · [optimizers](docs/concepts/optimizers.md) · [gate](docs/concepts/gate.md) · [serving](docs/concepts/serving.md)

## CLI reference

| Command | What it does |
|---|---|
| `promptline init` | Scaffold a commented `promptline.yaml` |
| `promptline demo setup [--offline]` | Build the support-assistant demo workspace |
| `promptline calibrate --gold <path\|helpsteer2>` | Certify the judge against human labels (κ threshold, saves certificate) |
| `promptline optimize --optimizer gepa [--budget N] [--data f.jsonl] [--resume id]` | Run an optimization pass; best candidate auto-registered |
| `promptline gate --candidate <id> --dev d.jsonl --val v.jsonl` | Statistically gate challengers vs the active prompt; promotes on win |
| `promptline registry list\|show\|activate\|rollback` | Inspect versions/lineage, bootstrap a baseline, undo a promotion |
| `promptline tui --run <id> \| --attach <sse-url>` | Live terminal cockpit for a run |
| `promptline serve [--host] [--port]` | Control plane + serving plane + dashboard over HTTP |
| `promptline data prepare --demo` | Alias forwarding to `demo setup` |

Exit codes follow the pipeline's semantics: `gate` returns 0 promote / 1 reject / 2 refusal.

## Dashboard & TUI

`promptline tui` is a Textual cockpit — score curve, per-example Pareto grid, candidate lineage, live event log with per-call cost, budget burn-down — attachable to live runs (SSE) or finished ones (events.jsonl). `promptline serve` also hosts the React dashboard (Runs, Lineage explorer, Judge calibration, Gate report, Registry).

Both share one design language: terminal-native in the style of opencode/Hermes — dark theme, monospace throughout, flat sharp-cornered panels with 1px borders, a single muted accent plus semantic green/red for verdicts, dense data-first layouts. No gradients, no glassmorphism.

## Testing

```bash
uv run pytest -q          # fully offline: FakeLLMClient scripts, exact stats tests
uv run ruff check .
```

The suite runs without network or keys: deterministic scripted LLM responses, statistical functions validated on known distributions, API via FastAPI TestClient. An opt-in live smoke test (needs `OPENROUTER_API_KEY`) runs a 5-example/10-rollout pass against a cheap model. All real LLM calls are cached in SQLite, so crashed runs resume cheaply and recorded traces replay cassette-style.

## License & citations

MIT.

- GEPA — Agrawal et al., [arXiv 2507.19457](https://arxiv.org/abs/2507.19457)
- MIPROv2 — Opsahl-Ong et al., [arXiv 2406.11695](https://arxiv.org/abs/2406.11695)
- ProTeGi / APO — Pryzant et al., EMNLP 2023, [arXiv 2305.03495](https://arxiv.org/abs/2305.03495)
- OPRO — Yang et al., [arXiv 2309.03409](https://arxiv.org/abs/2309.03409)
- CAPO (racing) — [arXiv 2504.16005](https://arxiv.org/abs/2504.16005)
- HelpSteer2 — Wang et al., [arXiv 2406.08673](https://arxiv.org/abs/2406.08673) (CC-BY-4.0)
- LLM-judge biases — Zheng et al., [arXiv 2306.05685](https://arxiv.org/abs/2306.05685)
- EvalGen (criteria drift) — Shankar et al., [arXiv 2404.12272](https://arxiv.org/abs/2404.12272)
