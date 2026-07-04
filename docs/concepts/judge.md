# Judges and calibration

`promptline/judge/` implements LLM-as-judge evaluation that is *measured before it is trusted*.

## Rubric judges

Both judges are built on `PromptProgram`, so a judge's instruction lives in a `Candidate` and can be optimized by Promptline's own optimizers (the meta-optimization pattern).

**`PointwiseJudge(criterion, judge_model, samples=1)`** scores one response on a `RubricCriterion` — a named criterion with a description, an integer `scale` (default `(1, 5)`), and optional per-point `anchors`. The generated instruction demands chain-of-thought first, verdict last (`[[reasoning]]:` then `[[score]]:`) and explicitly says "Do not reward length or verbosity". With `samples=1` it makes one temperature-0 call; with `samples>1` it makes k sampled calls (temperature 0.7, distinct seeds so a caching client sees distinct keys) and averages the parseable scores. `parse_score` extracts the first integer and clamps to the scale; if no sample parses, `JudgeError` is raised.

`judge.as_metric(client)` adapts the judge into a harness `Metric`: it scores the prediction's `answer`/`response` output, normalizes onto [0, 1] via `(v - lo) / (hi - lo)`, passes `labels["reference"]` as a reference when present, and **never raises** — judge failures come back as `MetricResult(score=0.0)`.

**`PairwiseJudge(criterion, judge_model)`** compares two responses with position debiasing: it judges *both orderings*; the verdict counts only when the two orderings agree (after un-swapping), otherwise the result is a `TIE` with both reasonings recorded.

## Known biases and mitigations

Following Zheng et al. (arXiv 2306.05685):

| Bias | Mitigation in Promptline |
|---|---|
| Position bias | Pairwise judge runs both orderings; disagreement → TIE |
| Verbosity bias | "Do not reward length" in the rubric; the gate's verbosity tripwire flags winners whose outputs are >1.5× longer |
| Self-preference | Config separates `models.judge` from `models.task`; the demo uses different families |
| Sampling noise | Temperature 0 by default, or k-sample averaging |
| One vague "overall" score | Per-criterion rubric judges (helpfulness, correctness, coherence, complexity, verbosity have built-in rubric text in the CLI) |

## Calibration

`Calibrator(judge, gold, client, threshold_kappa=0.6, label_range=None)` splits the gold dataset 50/50 into **dev** and **holdout** (deterministic content-hash split). Holdout is used only for certification; dev is the only data meta-optimization may see.

`await calibrator.calibrate()` judges every usable holdout record's `reference_output` and compares with the human labels:

- **Binning**: human scalar labels are mapped onto the judge's integer scale. If the observed (or declared, via `--label-min/--label-max`) range already equals the scale, the identity mapping is used; otherwise linear min-max rescaling + rounding (`binning="linear-minmax"`).
- **Metrics**: quadratic-weighted Cohen's κ on the binned labels (`promptline.judge.metrics.cohens_kappa`), Spearman ρ on the raw values, plus a full confusion matrix.
- **Degenerate guard**: if all binned human labels are identical, κ is meaningless — the certificate reports `degenerate=true` and fails.

## Certificates

The result is a `CalibrationCertificate` (criterion, κ, ρ, n_holdout, threshold, passed, confusion matrix, binning, label range, judge candidate id, timestamp), saved by the CLI to `<registry>/certificates/<criterion>.json` and listed by the server at `GET /judges/certificates`.

`require_certificate(path, min_kappa)` loads and validates one, raising `UncalibratedJudgeError` when the file is missing, `passed` is false, or κ is below the requirement. Set `gate.certificate` in `promptline.yaml` and the deploy gate refuses to run without a sufficient certificate.

```bash
promptline calibrate --gold gold.jsonl --criterion helpfulness --threshold 0.6 --label-min 0 --label-max 4
promptline calibrate --gold helpsteer2 --n 200        # pull gold data straight from HF
```

Exit code 1 when the certificate fails (it is still saved for inspection).

## Meta-optimizing the judge

`await calibrator.meta_optimize(optimizer, harness, budget)` optimizes the judge's own instruction on the **dev** half: examples are (conversation, response) pairs, the metric rewards closeness between the normalized judge score and the normalized human label (`score = 1 - |judge_norm - human_norm|`). The optimized judge candidate is then **re-certified on holdout**, so the certificate always reflects unseen data. Any registered optimizer works — the judge is just another `PromptProgram`.
