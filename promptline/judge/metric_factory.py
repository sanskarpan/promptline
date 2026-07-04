"""Build the optimization/gate metric from project config.

This is the wiring behind the headline chain *calibrated judge → optimizer →
gate*: :func:`resolve_metric` returns the calibrated-judge metric whenever the
``judge`` config section is enabled and a judge (or task) model is configured,
and falls back to the exact-match :func:`default_metric` otherwise.
"""
from __future__ import annotations

from pathlib import Path

from promptline.core.config import PromptlineConfig
from promptline.core.llm import LLMClient
from promptline.eval.harness import Metric, MetricResult
from promptline.judge.judge import PointwiseJudge, RubricCriterion

#: Default rubric descriptions per criterion name.
DEFAULT_RUBRICS: dict[str, str] = {
    "helpfulness": (
        "How well the response addresses the user's request and provides "
        "actionable, relevant information."
    ),
    "correctness": (
        "Whether the response is factually accurate and free of errors or "
        "unsupported claims."
    ),
    "coherence": (
        "Whether the response is well-structured, consistent, and easy to follow."
    ),
    "complexity": (
        "The intellectual depth required to write the response "
        "(domain expertise vs. basic language competency)."
    ),
    "verbosity": (
        "Whether the amount of detail is appropriate for the request — "
        "neither too terse nor padded."
    ),
}


def default_metric(example, prediction) -> MetricResult:  # type: ignore[no-untyped-def]
    """Exact-match on ``labels['answer']`` vs ``outputs.get('answer')``.

    The fallback metric used when the judge is disabled (``judge.enabled:
    false``) or no model is configured.
    """
    expected = example.labels.get("answer", "")
    got = prediction.outputs.get("answer", "")
    score = 1.0 if got.strip() == expected.strip() else 0.0
    return MetricResult(score=score, feedback=f"expected={expected!r} got={got!r}")


def rubric_from_config(cfg: PromptlineConfig) -> RubricCriterion:
    """Build the :class:`RubricCriterion` described by ``cfg.judge``."""
    criterion = cfg.judge.criterion
    description = cfg.judge.description or DEFAULT_RUBRICS.get(
        criterion, f"Rate the overall {criterion} of the response."
    )
    return RubricCriterion(
        name=criterion,
        description=description,
        scale=(cfg.judge.scale_min, cfg.judge.scale_max),
    )


def build_judge_metric(cfg: PromptlineConfig, client: LLMClient) -> Metric:
    """Construct a :class:`PointwiseJudge` metric from ``cfg.judge``.

    The judge model is ``cfg.models.judge``, falling back to the task model.
    References are read from ``labels['reference']`` when present.  Scores are
    normalized onto [0, 1].
    """
    judge_model = cfg.models.judge or cfg.models.task or "openai/gpt-4o-mini"
    judge = PointwiseJudge(criterion=rubric_from_config(cfg), judge_model=judge_model)
    return judge.as_metric(client)


def resolve_metric(cfg: PromptlineConfig, client: LLMClient) -> tuple[Metric, str]:
    """Return ``(metric, mode)`` where mode is ``"judge"`` or ``"exact-match"``.

    Judge mode requires ``cfg.judge.enabled`` and a configured judge or task
    model; anything else falls back to the exact-match metric.
    """
    if cfg.judge.enabled and (cfg.models.judge or cfg.models.task):
        return build_judge_metric(cfg, client), "judge"
    return default_metric, "exact-match"


def resolve_certificate_path(cfg: PromptlineConfig) -> Path:
    """Certificate path for the configured judge criterion.

    ``cfg.judge.certificate`` when set, else the default location written by
    ``promptline calibrate``: ``<registry>/certificates/<criterion>.json``.
    """
    if cfg.judge.certificate:
        return Path(cfg.judge.certificate)
    return Path(cfg.registry.path) / "certificates" / f"{cfg.judge.criterion}.json"
