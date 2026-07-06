"""Statistical deploy gate.

Compares challenger candidates against the incumbent prompt with paired
per-example statistics.  The gate REFUSES to run (raises) on undersized dev
sets, dev/val contamination, empty candidate lists, or — when configured — a
missing/insufficient judge calibration certificate.  A candidate is promoted
only when it survives Holm-corrected significance testing on dev AND its
paired confidence interval on the held-out val set excludes zero.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from promptline.core.config import GateConfig, JudgeConfig
from promptline.core.program import PromptProgram
from promptline.core.types import Candidate, Example
from promptline.eval.harness import Budget, EvalHarness, Metric
from promptline.eval.stats import (
    bootstrap_pvalue,
    holm_correct,
    min_examples_warning,
    paired_bootstrap_ci,
)
from promptline.judge.calibrator import require_certificate

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class GateSettings:
    """Local gate settings; ``alpha``/``min_examples`` mirror core GateConfig."""

    alpha: float = 0.05
    min_examples: int = 50
    verbosity_ratio_flag: float = 1.5
    n_spot_samples: int = 5
    require_certificate_path: Path | None = None
    min_kappa: float = 0.6

    @classmethod
    def from_config(cls, cfg: GateConfig, judge: JudgeConfig | None = None) -> GateSettings:
        """Build settings from the ``gate`` (and optionally ``judge``) config.

        ``judge.certificate`` is the primary certificate location;
        ``gate.certificate`` is honored for back-compat and wins when set.
        When the judge section supplies the certificate, its ``min_kappa`` is
        used too.
        """
        certificate = cfg.certificate
        min_kappa = cfg.min_kappa
        if not certificate and judge is not None and judge.certificate:
            certificate = judge.certificate
            min_kappa = judge.min_kappa
        return cls(
            alpha=cfg.alpha,
            min_examples=cfg.min_examples,
            require_certificate_path=Path(certificate) if certificate else None,
            min_kappa=min_kappa,
        )


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------


class CandidateGateResult(BaseModel):
    """Paired dev-set statistics for one challenger vs the incumbent."""

    candidate_id: str
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    holm_significant: bool
    dev_mean: float
    incumbent_dev_mean: float


class GateReport(BaseModel):
    """Full outcome of a gate run; JSON round-trippable."""

    program: str
    incumbent_id: str
    results: list[CandidateGateResult]
    winner_id: str | None
    val_mean_delta: float | None = None
    val_ci_low: float | None = None
    val_ci_high: float | None = None
    verdict: Literal["promote", "reject"]
    flags: list[str] = []
    spot_samples: list[dict] = []
    warnings: list[str] = []
    created_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _example_hash(example: Example) -> str:
    """Content hash over inputs+labels for dev/val overlap detection."""
    canonical = json.dumps({"inputs": example.inputs, "labels": example.labels}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


async def _direct_eval(
    program: PromptProgram,
    candidate: Candidate,
    examples: list[Example],
    harness: EvalHarness,
    metric: Metric,
    budget: Budget | None,
) -> list[tuple[int, str, float]]:
    """Sequentially run *candidate* over *examples* via ``program.run``.

    Returns ``(example_idx, joined_output_text, score)`` per example so the
    caller can inspect raw outputs (verbosity tripwire, spot samples), which
    :class:`EvalHarness` does not expose.  Rollouts are charged against
    *budget* via ``try_reserve``; exhaustion truncates the run.
    """
    results: list[tuple[int, str, float]] = []
    for idx, example in enumerate(examples):
        if budget is not None and not await budget.try_reserve(rollouts=1):
            break
        try:
            prediction = await program.run(example, candidate, harness.client, harness.cfg)
        except Exception:
            results.append((idx, "", 0.0))
            continue
        if budget is not None:
            budget.add_cost(prediction.cost_usd)
        if prediction.failed:
            results.append((idx, "", 0.0))
            continue
        raw = metric(example, prediction)
        if inspect.isawaitable(raw):
            raw = await raw
        output = "\n".join(prediction.outputs.values())
        results.append((idx, output, float(raw.score)))
    return results


def _paired_deltas(
    candidate_scores: dict[int, float], incumbent_scores: dict[int, float]
) -> list[float]:
    """Deltas over example indices present in BOTH runs (truncation-safe)."""
    shared = sorted(candidate_scores.keys() & incumbent_scores.keys())
    return [candidate_scores[i] - incumbent_scores[i] for i in shared]


def _mean_len(outputs: list[str]) -> float:
    if not outputs:
        return 0.0
    return sum(len(o) for o in outputs) / len(outputs)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


async def run_gate(
    program: PromptProgram,
    incumbent: Candidate,
    candidates: list[Candidate],
    dev: list[Example],
    val: list[Example],
    harness: EvalHarness,
    metric: Metric,
    settings: GateSettings,
    budget: Budget | None = None,
    collect_outputs: bool = True,
) -> GateReport:
    """Statistically gate *candidates* against *incumbent*.

    Procedure
    ---------
    1. Refusals: undersized dev set, dev/val contamination, empty candidate
       list (``ValueError``); missing/weak calibration certificate when
       ``settings.require_certificate_path`` is set
       (``UncalibratedJudgeError``).
    2. Dev phase: evaluate incumbent once and each candidate on the same dev
       examples via *harness*; per-candidate paired deltas feed a bootstrap
       p-value and CI.
    3. Holm-correct all candidate p-values; survivors must also have positive
       mean delta.  Winner = survivor with the largest mean delta; when none
       survives the verdict is ``reject`` and val evaluation is skipped.
    4. Val confirmation: paired winner-vs-incumbent deltas on val; promote
       only when the CI excludes zero (``ci_low > 0``).
    5. Tripwires (never block): the verbosity flag and spot samples require
       raw model outputs, which :class:`EvalHarness` does not return.  With
       ``collect_outputs=True`` (default) the val phase therefore runs through
       a direct ``program.run`` loop (budget-charged via ``try_reserve``)
       instead of the harness; dev-phase evaluations always use the harness.
       With ``collect_outputs=False`` the val phase also uses the harness and
       verbosity/spot samples are skipped.
    """
    if not candidates:
        raise ValueError("no candidates to gate; provide at least one challenger")
    if len(dev) < settings.min_examples:
        warning = min_examples_warning(len(dev), floor=settings.min_examples)
        raise ValueError(
            f"dev set too small for gating ({len(dev)} < {settings.min_examples} "
            f"examples). {warning}"
        )
    dev_hashes = {_example_hash(e) for e in dev}
    val_hashes = {_example_hash(e) for e in val}
    overlap = dev_hashes & val_hashes
    if overlap:
        raise ValueError(
            f"dev/val contamination: {len(overlap)} example(s) appear in both "
            "splits; gating on contaminated data is invalid"
        )
    if settings.require_certificate_path is not None:
        require_certificate(settings.require_certificate_path, settings.min_kappa)

    warnings: list[str] = []
    val_warning = min_examples_warning(len(val), floor=settings.min_examples)
    if val_warning is not None:
        warnings.append(val_warning)

    program_name = program.modules[0].name if program.modules else ""
    created_at = datetime.now(UTC).isoformat()

    # ---- Dev phase (harness) ---------------------------------------------
    incumbent_dev = await harness.evaluate(program, incumbent, dev, metric, budget)
    incumbent_scores = {r.example_idx: r.score for r in incumbent_dev.per_example}

    results: list[CandidateGateResult] = []
    pvals: list[float] = []
    for candidate in candidates:
        report = await harness.evaluate(program, candidate, dev, metric, budget)
        cand_scores = {r.example_idx: r.score for r in report.per_example}
        deltas = _paired_deltas(cand_scores, incumbent_scores)
        if deltas:
            p_value = bootstrap_pvalue(deltas)
            mean_delta, ci_low, ci_high = paired_bootstrap_ci(deltas, alpha=settings.alpha)
        else:
            warnings.append(f"candidate {candidate.id}: no paired dev examples; skipped")
            p_value, mean_delta, ci_low, ci_high = 1.0, 0.0, 0.0, 0.0
        pvals.append(p_value)
        results.append(
            CandidateGateResult(
                candidate_id=candidate.id,
                mean_delta=mean_delta,
                ci_low=ci_low,
                ci_high=ci_high,
                p_value=p_value,
                holm_significant=False,
                dev_mean=report.mean_score,
                incumbent_dev_mean=incumbent_dev.mean_score,
            )
        )

    # ---- Holm correction ---------------------------------------------------
    for result, significant in zip(results, holm_correct(pvals, alpha=settings.alpha), strict=True):
        result.holm_significant = significant

    survivors = [r for r in results if r.holm_significant and r.mean_delta > 0]
    if not survivors:
        return GateReport(
            program=program_name,
            incumbent_id=incumbent.id,
            results=results,
            winner_id=None,
            verdict="reject",
            warnings=warnings,
            created_at=created_at,
        )

    winner_result = max(survivors, key=lambda r: r.mean_delta)
    winner = next(c for c in candidates if c.id == winner_result.candidate_id)

    # ---- Val confirmation ---------------------------------------------------
    flags: list[str] = []
    spot_samples: list[dict] = []
    if collect_outputs:
        incumbent_val = await _direct_eval(program, incumbent, val, harness, metric, budget)
        winner_val = await _direct_eval(program, winner, val, harness, metric, budget)
        incumbent_val_scores = {i: s for i, _, s in incumbent_val}
        winner_val_scores = {i: s for i, _, s in winner_val}

        incumbent_mean_len = _mean_len([o for _, o, _ in incumbent_val])
        winner_mean_len = _mean_len([o for _, o, _ in winner_val])
        if (
            incumbent_mean_len > 0
            and winner_mean_len > settings.verbosity_ratio_flag * incumbent_mean_len
        ):
            flags.append("verbosity")
        spot_samples = [
            {"example_idx": i, "output": o, "score": s}
            for i, o, s in winner_val[: settings.n_spot_samples]
        ]
    else:
        incumbent_val_report = await harness.evaluate(program, incumbent, val, metric, budget)
        winner_val_report = await harness.evaluate(program, winner, val, metric, budget)
        incumbent_val_scores = {r.example_idx: r.score for r in incumbent_val_report.per_example}
        winner_val_scores = {r.example_idx: r.score for r in winner_val_report.per_example}

    val_deltas = _paired_deltas(winner_val_scores, incumbent_val_scores)
    if not val_deltas:
        warnings.append("no paired val examples; cannot confirm winner")
        return GateReport(
            program=program_name,
            incumbent_id=incumbent.id,
            results=results,
            winner_id=winner.id,
            verdict="reject",
            flags=flags,
            spot_samples=spot_samples,
            warnings=warnings,
            created_at=created_at,
        )

    paired_val_n = len(val_deltas)
    if paired_val_n < settings.min_examples:
        warnings.append(f"val truncated to n={paired_val_n} (< min_examples)")

    val_mean_delta, val_ci_low, val_ci_high = paired_bootstrap_ci(val_deltas, alpha=settings.alpha)
    verdict: Literal["promote", "reject"] = "promote" if val_ci_low > 0 else "reject"
    if paired_val_n < 10:
        flags.append("val_too_small")
        verdict = "reject"
    return GateReport(
        program=program_name,
        incumbent_id=incumbent.id,
        results=results,
        winner_id=winner.id,
        val_mean_delta=val_mean_delta,
        val_ci_low=val_ci_low,
        val_ci_high=val_ci_high,
        verdict=verdict,
        flags=flags,
        spot_samples=spot_samples,
        warnings=warnings,
        created_at=created_at,
    )
