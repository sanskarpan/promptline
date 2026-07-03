"""Judge calibration against gold human labels.

A :class:`Calibrator` splits a gold dataset 50/50 into dev/holdout.  The
holdout half is used only for certification (:meth:`Calibrator.calibrate`);
the dev half is the only data an optimizer sees during
:meth:`Calibrator.meta_optimize`.

Binning ("linear-minmax"): human scalar labels are linearly rescaled from
their observed min/max onto the judge's integer scale and rounded.  When the
observed human label range already equals the judge scale, the identity
mapping is used (binning="identity").
"""
from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from promptline.core.llm import LLMClient
from promptline.core.types import Candidate, Example
from promptline.data.dataset import Dataset, Record
from promptline.eval.harness import Budget, EvalHarness, MetricResult
from promptline.judge.judge import PointwiseJudge, render_transcript
from promptline.judge.metrics import cohens_kappa, spearman
from promptline.optimizers.base import Optimizer

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class UncalibratedJudgeError(Exception):
    """Raised when a required calibration certificate is missing or failed."""


# ---------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------


class CalibrationCertificate(BaseModel):
    """Evidence that a judge agrees with human labels on a held-out set."""

    judge_name: str
    criterion: str
    kappa: float
    spearman: float
    n_holdout: int
    threshold: float
    passed: bool
    judge_candidate_id: str
    created_at: str
    confusion: list[list[int]]
    binning: str
    degenerate: bool = False
    label_min: float = 0.0
    label_max: float = 0.0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CalibrationCertificate:
        return cls.model_validate_json(Path(path).read_text())


def require_certificate(
    cert_path: str | Path,
    min_kappa: float = 0.6,
) -> CalibrationCertificate:
    """Load a certificate, raising :class:`UncalibratedJudgeError` unless valid.

    Raises when the file is missing, the certificate did not pass, or its
    kappa is below *min_kappa*.
    """
    path = Path(cert_path)
    if not path.exists():
        raise UncalibratedJudgeError(
            f"no calibration certificate at {path}; run `promptline calibrate` first"
        )
    cert = CalibrationCertificate.load(path)
    if not cert.passed or cert.kappa < min_kappa:
        raise UncalibratedJudgeError(
            f"certificate at {path} is not sufficient "
            f"(passed={cert.passed}, kappa={cert.kappa:.3f}, required>={min_kappa})"
        )
    return cert


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


def _usable(record: Record) -> bool:
    """A record is usable when it has a judged response and a scalar label."""
    return record.reference_output is not None and isinstance(
        record.human_label, (int, float)
    )


class Calibrator:
    """Certifies (and optionally meta-optimizes) a pointwise judge."""

    def __init__(
        self,
        judge: PointwiseJudge,
        gold: Dataset,
        client: LLMClient,
        threshold_kappa: float = 0.6,
        split_seed: int = 0,
        label_range: tuple[float, float] | None = None,
    ) -> None:
        self.judge = judge
        self.client = client
        self.threshold_kappa = threshold_kappa
        self._label_range = label_range
        splits = gold.split({"dev": 0.5, "holdout": 0.5}, seed=split_seed)
        self.dev = splits["dev"]
        self.holdout = splits["holdout"]

    # ------------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------------

    def _bin_human_labels(
        self, values: list[float]
    ) -> tuple[list[int], str, float, float]:
        """Map human scalar labels onto the judge's integer scale.

        Returns ``(binned, binning_name, vmin, vmax)`` where *vmin*/*vmax* are
        the range actually used for binning (declared or observed).
        """
        lo, hi = self.judge.criterion.scale
        if self._label_range is not None:
            vmin, vmax = self._label_range
        else:
            vmin, vmax = min(values), max(values)
        if vmin == lo and vmax == hi:
            return [round(v) for v in values], "identity", vmin, vmax
        if vmax == vmin:
            mid = round((lo + hi) / 2)
            return [mid] * len(values), "linear-minmax", vmin, vmax
        binned = [
            round(lo + (v - vmin) / (vmax - vmin) * (hi - lo)) for v in values
        ]
        return binned, "linear-minmax", vmin, vmax

    # ------------------------------------------------------------------
    # Certification
    # ------------------------------------------------------------------

    async def calibrate(
        self,
        candidate: Candidate | None = None,
    ) -> CalibrationCertificate:
        """Judge every holdout record's reference output and certify agreement."""
        records = [r for r in self.holdout if _usable(r)]
        if not records:
            raise ValueError(
                "no usable holdout records: each needs reference_output and a "
                "numeric human_label"
            )

        human_raw = [float(r.human_label) for r in records]  # type: ignore[arg-type]
        judge_raw: list[float] = []
        for record in records:
            judged = await self.judge.score(
                record,
                record.reference_output,  # type: ignore[arg-type]
                self.client,
                candidate,
            )
            judge_raw.append(judged.value)

        lo, hi = self.judge.criterion.scale
        human_binned, binning, label_min, label_max = self._bin_human_labels(human_raw)
        judge_binned = [max(lo, min(hi, round(v))) for v in judge_raw]

        rho = spearman(human_raw, judge_raw)
        if math.isnan(rho):
            rho = 0.0

        # Degenerate check: kappa is not meaningful when all binned human
        # labels are identical (< 2 distinct values).
        degenerate = len(set(human_binned)) < 2
        if degenerate:
            kappa = 0.0
            passed = False
        else:
            kappa = cohens_kappa(human_binned, judge_binned, weights="quadratic")
            passed = kappa >= self.threshold_kappa

        size = hi - lo + 1
        confusion = [[0] * size for _ in range(size)]
        for h, j in zip(human_binned, judge_binned, strict=True):
            confusion[h - lo][j - lo] += 1

        used = candidate or self.judge.seed_candidate
        return CalibrationCertificate(
            judge_name=f"pointwise:{self.judge.judge_model}",
            criterion=self.judge.criterion.name,
            kappa=kappa,
            spearman=rho,
            n_holdout=len(records),
            threshold=self.threshold_kappa,
            passed=passed,
            judge_candidate_id=used.id,
            created_at=datetime.now(UTC).isoformat(),
            confusion=confusion,
            binning=binning,
            degenerate=degenerate,
            label_min=label_min,
            label_max=label_max,
        )

    # ------------------------------------------------------------------
    # Meta-optimization of the judge prompt (dev split only)
    # ------------------------------------------------------------------

    async def meta_optimize(
        self,
        optimizer: Optimizer,
        harness: EvalHarness,
        budget: Budget,
    ) -> tuple[Candidate, CalibrationCertificate]:
        """Optimize the judge instruction on DEV; re-certify on HOLDOUT."""
        records = [r for r in self.dev if _usable(r)]
        if not records:
            raise ValueError(
                "no usable dev records: each needs reference_output and a "
                "numeric human_label"
            )

        trainset = [
            Example(
                inputs={
                    "conversation": render_transcript(r),
                    "response": r.reference_output,  # type: ignore[dict-item]
                },
                labels={"human_score": str(float(r.human_label))},  # type: ignore[arg-type]
            )
            for r in records
        ]

        lo, hi = self.judge.criterion.scale
        human_values = [float(r.human_label) for r in records]  # type: ignore[arg-type]
        hmin, hmax = min(human_values), max(human_values)

        def metric(example: Example, prediction) -> MetricResult:
            value = self.judge.parse_score(prediction.outputs.get("score", ""))
            if value is None:
                return MetricResult(score=0.0, feedback="unparseable judge score")
            judge_norm = (value - lo) / (hi - lo)
            human = float(example.labels["human_score"])
            human_norm = 0.5 if hmax == hmin else (human - hmin) / (hmax - hmin)
            error = abs(judge_norm - human_norm)
            return MetricResult(
                score=1.0 - error,
                feedback=f"judge={value} human={human} abs_err={error:.3f}",
            )

        result = await optimizer.optimize(
            program=self.judge.program,
            seed=self.judge.seed_candidate,
            trainset=trainset,
            metric=metric,
            budget=budget,
            harness=harness,
        )
        certificate = await self.calibrate(result.best)
        return result.best, certificate
