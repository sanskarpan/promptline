# Optimizers

All optimizers implement one contract (`promptline.optimizers.base.Optimizer`):

```python
async def optimize(program, seed, trainset, metric, budget, harness,
                   emit=lambda e: None) -> OptimizeResult
```

`OptimizeResult` holds `best`, all `candidates` (with lineage via `parent_ids`), a `scores` dict, and `events_count`. The metric follows GEPA's richer signature — it returns `MetricResult(score, feedback, per_module)`, so optimizers can learn from *textual* feedback, not just numbers.

## Budget semantics

`Budget(max_rollouts, max_cost_usd)` is a hard wall enforced centrally:

- Every program execution costs one rollout, reserved atomically with `await budget.try_reserve(rollouts=1)` *before* the call; costs are charged after (`add_cost`).
- `EvalHarness.evaluate` never raises on exhaustion — it truncates and sets `EvalReport.truncated`. Optimizer loops check `budget.exhausted` (or catch `BudgetExhausted` in their own loops) and end gracefully with best-so-far.
- Reflection/proposal/paraphrase LLM calls charge cost but not rollouts.

## Run events

Optimizers emit typed `RunEvent`s (`run_started`, `candidate_proposed`, `minibatch_scored`, `full_eval`, `pareto_updated`, `merge_attempted`, `budget_tick`, `run_finished`) consumed by the TUI and dashboard via SSE. The CLI persists them with `RunRecorder` to `<registry>/runs/<run_id>/events.jsonl`; GEPA carries its own recorder plus a `checkpoint.json` written after every accepted candidate (Ctrl-C is a clean checkpoint; `--resume <run_id>` continues, with the caveat that RNG state is not checkpointed).

## GEPA — `--optimizer gepa` (flagship)

Genetic-Pareto reflective evolution (arXiv 2507.19457). Defaults: minibatch b=3, `n_pareto=32` (capped at half the trainset), merges every 5 acceptances.

```
split trainset -> D_pareto (selection) + D_feedback (reflection)
pool = {seed}; S[seed] = per-instance scores on D_pareto
loop until budget wall:
    parent = pareto_sample(S)            # Algorithm 2
    run parent on minibatch of D_feedback, keep traces + feedback
    new_instruction = reflect(traces, feedback)   # reflection model
    child = parent.child(mutated module)
    if child minibatch score > parent's:          # strict acceptance
        full-eval child on D_pareto; add to pool; checkpoint
    every merge_every acceptances: system-aware merge of two
        unrelated frontier candidates via their common ancestor
```

Selection (`gepa/pareto.py`) keeps the **per-instance Pareto frontier**: every candidate that is best on at least one `D_pareto` instance survives (dominated ones pruned), and sampling weight is proportional to how many instances a candidate wins. This preserves "specialist" stepping stones a greedy mean would discard. Merges (`gepa/merge.py`, Appendix F) recombine module-by-module with the triplet rule: the parent that diverged from the common ancestor carries the learned mutation.

## MIPRO — `--optimizer mipro`

Three stages (arXiv 2406.11695): (1) bootstrap demo sets per module (set 0 always empty/zero-shot), reusing `collect_demo_pool`; (2) grounded instruction proposal — one LLM call summarizes the dataset, a programmatic program summary is built, then per-module proposals are generated with history and a randomized style tip; (3) an Optuna TPE study over the categorical space `inst_<module> × demo_<module>`, each trial scored on a fresh seeded minibatch (default 16), with a full evaluation of the best pending config every `full_eval_steps=5` trials and once at the end.

## BootstrapFewShot / BootstrapRandomSearch — `--optimizer bootstrap` / `bootstrap-rs`

The cheap tier. `BootstrapFewShot` runs the seed as its own teacher over the trainset; predictions whose metric score reaches `threshold` (default 1.0) become per-module `Demo`s (up to `max_demos=4`), attached to a single child candidate. `BootstrapRandomSearch` splits off a validation fraction, collects a larger demo pool, then scores `n_subsets` random demo subsets on the val split and keeps the winner. No instructions are edited — demos only.

## ProTeGi — `--optimizer protegi`

Textual gradients with racing (arXiv 2305.03495 + CAPO arXiv 2504.16005). Defaults: beam 4, 3 rounds. Per round, for each beam candidate: score a minibatch → feed up to 4 failing examples to the LLM for a 2-3 sentence critique (the "gradient") → a second call rewrites the instruction to fix the diagnosis → paraphrase calls expand each edit. Parents + children are pruned back to the beam with successive-halving racing: survivors are scored on fresh racing batches and the bottom half is dropped each racing round — far cheaper than full-evaluating every candidate. This is the "reflection without evolution" ablation against GEPA.

## OPRO — `--optimizer opro`

The ~200-line baseline (arXiv 2309.03409). Each step builds a meta-prompt containing the (instruction, score) trajectory sorted ascending (best last) and asks the reflection model for a better instruction inside `<INS>` tags; proposals are scored on the trainset (or a minibatch) and appended to the trajectory. Documented caveat: extrapolating a score trajectory requires a strong proposer model (see arXiv 2405.10276) — with weak proposers, prefer GEPA/ProTeGi, which reason over *failures* rather than score sequences.
