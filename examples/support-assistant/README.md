# Support-Assistant Demo

The end-to-end Promptline story on a customer-support task:

1. **Setup** — build datasets (HelpSteer2 gold labels + Bitext support conversations) and a config with a deliberately mediocre seed prompt.
2. **Calibrate** — certify a helpfulness judge against human labels.
3. **Optimize** — evolve the seed prompt with GEPA under a hard $5 budget.
4. **Gate & serve** — statistically gate the winner, promote it, and `curl` the serving endpoint.

You bring your own [OpenRouter](https://openrouter.ai/) API key; total cost for the full demo is a few dollars on the default cheap models.

## Step 0 — install

```bash
uv pip install -e ".[data]"          # the data extra pulls HF `datasets`
export OPENROUTER_API_KEY=sk-or-...
```

## Step 1 — setup

```bash
promptline demo setup                # writes examples/support-assistant/workspace/
cd examples/support-assistant/workspace
```

This downloads HelpSteer2 (400 rows, per-attribute human ratings, CC-BY-4.0) and the Bitext customer-support dataset, then writes:

| File | Contents |
|---|---|
| `gold.jsonl` | Judge gold set: conversations + responses + 0–4 human helpfulness labels |
| `dev.jsonl` / `val.jsonl` | Seeded, hash-disjoint task splits for the gate |
| `feedback.jsonl` | Remaining task rows (the optimizer's playground) |
| `train.jsonl` | `feedback` rows in optimize format (`inputs.conversation` / `labels.reference`) |
| `promptline.yaml` | Seed prompt "You are a support agent. Answer the question.", cheap task model, 300-rollout / $5 budget |

Tune sizes with `--gold-n`, `--dev-n`, `--val-n`.

## Step 2 — calibrate the judge

```bash
promptline calibrate --gold gold.jsonl --label-min 0 --label-max 4
```

This scores every holdout gold response with the `anthropic/claude-3.5-haiku` rubric judge and compares against the human labels. Expect output like:

```
        Calibration Certificate
 criterion              helpfulness
 kappa (quadratic)            0.68
 spearman                     0.71
 n_holdout                     ~200
 threshold                    0.60
 binning              linear-minmax
 passed                        yes
Certificate saved to .promptline/certificates/helpfulness.json
```

For context, GPT-4 as an MT-Bench judge reaches ~80% agreement with humans (arXiv 2306.05685); quadratic-weighted κ ≥ 0.6 is "substantial" agreement. If calibration fails, the certificate is still saved (with `passed: false`) so you can inspect the confusion matrix, but the exit code is 1 — and `optimize`/`gate` will *refuse to run* against a missing or failed certificate, so this step genuinely unlocks the rest of the chain.

The demo config points `judge.certificate` at exactly the path this command writes (`.promptline/certificates/helpfulness.json`), so nothing else needs wiring.

## Step 3 — optimize with GEPA

```bash
promptline optimize --optimizer gepa --data train.jsonl
```

Candidates are scored by the calibrated helpfulness judge from step 2 — look for the `Metric: judge(helpfulness)` line at startup. Without a passing certificate this exits with code 2 (`--allow-uncalibrated` bypasses the check, loudly, if you really must).

Watch it live from another terminal — the run id is printed at start:

```bash
promptline tui --run <run_id>        # score curve, Pareto grid, lineage, budget burn-down
# or the web dashboard:
promptline serve                     # then open http://127.0.0.1:8000
```

What to watch for:

- `candidate_proposed` / `minibatch_scored` events as GEPA reflects on failing traces and mutates the instruction;
- the per-instance **Pareto frontier** growing — specialists that win on some examples survive even if their mean is lower;
- occasional `merge_attempted` events combining two lineages;
- `budget_tick` counting down the 300-rollout / $5 wall. The run ends gracefully with best-so-far when the budget is hit.

The final table shows candidates by score; the best is auto-registered in `.promptline/registry.db`.

## Step 4 — gate, promote, serve

```bash
# Bootstrap a baseline: activate the seed (worst) prompt from `registry list`
promptline registry list
promptline registry activate <seed_prompt_id>

# Challenge it with the GEPA winner
promptline gate --candidate <best_prompt_id> --dev dev.jsonl --val val.jsonl
```

The gate computes paired per-example deltas, a 10k-resample bootstrap CI, and Holm-corrected p-values on `dev`, then confirms the winner on the untouched `val` split. On `promote`, the registry's active pointer advances. Then:

```bash
promptline serve &
curl -s http://127.0.0.1:8000/prompts/support/active | python -m json.tool
```

```json
{
  "program": "support",
  "prompt_id": "…",
  "modules": {"support": {"instruction": "…the evolved prompt…", "demos": []}},
  "activated_at": "…"
}
```

The endpoint sends an `ETag`; poll with `If-None-Match` to get cheap `304`s until the next promotion. Regret a promotion? `promptline registry rollback`.

## Offline dry run (no API key, no downloads)

Everything above can be rehearsed offline with bundled fixtures and a scripted fake LLM:

```bash
promptline demo setup --offline --dir /tmp/promptline-demo
cd /tmp/promptline-demo

# Point the CLI at a fake-response script instead of OpenRouter:
# keyed rule answers judge prompts with a score, everything else with an answer.
cat > fake.json <<'EOF'
{"keyed": [{"contains": "impartial expert evaluator", "response": "[[reasoning]]: fine\n[[score]]: 3"}],
 "responses": ["[[answer]]: Sure — here is how to do that."]}
EOF
export PROMPTLINE_FAKE_SCRIPT=$PWD/fake.json

promptline calibrate --gold gold.jsonl --label-min 0 --label-max 4   # runs; κ will be low (constant judge)
# The constant fake judge fails calibration, so optimize refuses to run
# against it — exactly the production behavior. For the offline rehearsal,
# bypass the certificate check explicitly:
promptline optimize --optimizer bootstrap --data train.jsonl --allow-uncalibrated
promptline registry list
```

`PROMPTLINE_FAKE_SCRIPT` points at a JSON file with a cycling `"responses"` list and optional `"keyed"` rules (`[{"contains": ..., "response": ...}]`) matched against the prompt text — enough to exercise the whole pipeline deterministically, which is exactly how the test suite does it.

Two size caveats in offline mode (the bundled fixtures are deliberately tiny — 50 gold rows, 60 support rows):

- **Calibration:** the calibrator holds out half the gold set, so the certificate reports `n_holdout ≈ 25` instead of the ~200 you get online. κ is computed the same way, just on a much smaller (noisier) holdout.
- **Gating:** the offline val split is only ~15 rows (dev ~18), which is below the gate's `min_examples: 50` — `promptline gate` will refuse to run on the offline splits. Use the online datasets for a real gate run, or lower `gate.min_examples` in `promptline.yaml` (e.g. to 10) for a toy rehearsal; at these sizes a `val_too_small` reject remains possible by design.
