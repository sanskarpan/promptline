# The deploy gate

`promptline.registry.gate.run_gate` decides whether a challenger prompt replaces the incumbent. It is deliberately paranoid: a candidate is promoted only when it survives Holm-corrected significance testing on the dev split **and** its confidence interval on an untouched validation split excludes zero.

```bash
promptline gate --candidate <id> [--candidate <id> ...] --dev dev.jsonl --val val.jsonl
# exit codes: 0 promote · 1 reject · 2 refusal
```

## Refusals (the gate won't even run)

- Dev set smaller than `gate.min_examples` (default 50) → `ValueError` with a power warning.
- **Dev/val contamination**: content hashes over `inputs+labels` of the two splits must be disjoint.
- Empty candidate list.
- When `gate.certificate` is configured: a missing, failed, or under-κ calibration certificate → `UncalibratedJudgeError` (see [judge.md](judge.md)).

A small *val* set only warns; a small dev set refuses.

## Dev phase: paired bootstrap + Holm

The incumbent is evaluated once on dev; each candidate is evaluated on the *same* examples. For each candidate the per-example paired deltas `d_i = s_cand,i − s_inc,i` (over indices present in both runs, so budget truncation stays paired) feed two statistics from `promptline.eval.stats`:

**Percentile bootstrap CI** (`paired_bootstrap_ci`, 10,000 resamples, seeded): resample the deltas with replacement, take the mean of each resample, and report the `α/2` and `1−α/2` percentiles around the observed mean.

**Bootstrap p-value** (`bootstrap_pvalue`): center the deltas at zero (the null world), resample, and count how often the null mean is at least as extreme as the observed one:

```
p = (1 + #{ |mean(boot)| ≥ |mean(obs)| }) / (n_boot + 1)
```

**Holm–Bonferroni** (`holm_correct`) then controls the family-wise error rate across multiple candidates: sort p-values ascending (smallest p first) and compare the k-th against `α / (n − k + 1)` for k = 1..n, stopping at the first failure. Survivors must *also* have a positive mean delta. The winner is the survivor with the largest mean delta; if none survives, the verdict is `reject` and val is never touched (no information leaks into it).

## Val confirmation

The winner and incumbent are re-run on the held-out val split; promotion requires the paired bootstrap CI on val to exclude zero (`ci_low > 0`). Fewer than 10 paired val examples forces `reject` with a `val_too_small` flag; a val set under `min_examples` adds a warning.

## Tripwires (flag, never block)

Judge-hacking detectors on the winner's raw val outputs:

- **Verbosity flag**: mean output length > 1.5× the incumbent's (`verbosity_ratio_flag`) — the classic length exploit against LLM judges.
- **Spot samples**: the first `n_spot_samples=5` val outputs are embedded in the report for human review.

Collecting raw outputs requires bypassing the harness, so with `collect_outputs=True` (default) the val phase runs a direct sequential `program.run` loop (still budget-charged); dev always uses the concurrent harness.

## Report and promotion

The `GateReport` (per-candidate deltas/CIs/p-values, winner, val CI, verdict, flags, spot samples, warnings) is JSON round-trippable; the CLI saves it under `<registry>/gate_reports/` and — on `promote` — calls `registry.activate(program, winner_id, report_json)`, the only way the active pointer moves forward. `GateSettings.from_config` maps the `gate:` section of `promptline.yaml` (`alpha`, `min_examples`, `certificate`, `min_kappa`) onto the gate.
