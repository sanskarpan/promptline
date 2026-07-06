"""Tests for promptline.judge.metric_factory — judge-as-metric wiring."""

from __future__ import annotations

from pathlib import Path

from promptline.core.config import PromptlineConfig
from promptline.core.llm import FakeLLMClient
from promptline.core.program import Prediction
from promptline.core.types import Example
from promptline.judge.metric_factory import (
    build_judge_metric,
    default_metric,
    resolve_certificate_path,
    resolve_metric,
    rubric_from_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> PromptlineConfig:
    base = {
        "models": {"task": "fake/task", "judge": "fake/judge"},
        "registry": {"path": ".reg"},
    }
    base.update(overrides)
    return PromptlineConfig.model_validate(base)


def _prediction(outputs: dict[str, str]) -> Prediction:
    return Prediction(outputs=outputs, traces=[], cost_usd=0.0)


def _judge_resp(score: str) -> str:
    return f"[[reasoning]]: fine\n[[score]]: {score}"


# ---------------------------------------------------------------------------
# default_metric (exact match)
# ---------------------------------------------------------------------------


def test_default_metric_exact_match() -> None:
    example = Example(inputs={}, labels={"answer": "yes"})
    assert default_metric(example, _prediction({"answer": " yes "})).score == 1.0
    assert default_metric(example, _prediction({"answer": "no"})).score == 0.0


# ---------------------------------------------------------------------------
# build_judge_metric
# ---------------------------------------------------------------------------


async def test_build_judge_metric_returns_normalized_score() -> None:
    cfg = _cfg()
    client = FakeLLMClient(script=[_judge_resp("5")])
    metric = build_judge_metric(cfg, client)
    example = Example(inputs={"conversation": "user: hi"}, labels={"reference": "ref"})
    result = await metric(example, _prediction({"answer": "hello"}))
    assert result.score == 1.0  # 5 on a 1-5 scale → 1.0
    # Judge model from config is used in the call.
    assert client.calls[0].model == "fake/judge"


async def test_build_judge_metric_midpoint_score_and_custom_scale() -> None:
    cfg = _cfg(judge={"criterion": "helpfulness", "scale_min": 1, "scale_max": 3})
    client = FakeLLMClient(script=[_judge_resp("2")])
    metric = build_judge_metric(cfg, client)
    result = await metric(
        Example(inputs={"conversation": "user: hi"}),
        _prediction({"answer": "hello"}),
    )
    assert result.score == 0.5  # 2 on a 1-3 scale


def test_rubric_from_config_uses_default_description() -> None:
    rubric = rubric_from_config(_cfg())
    assert rubric.name == "helpfulness"
    assert "user's request" in rubric.description

    custom = rubric_from_config(_cfg(judge={"criterion": "brandvoice"}))
    assert "brandvoice" in custom.description

    explicit = rubric_from_config(
        _cfg(judge={"criterion": "helpfulness", "description": "CUSTOM-DESC"})
    )
    assert explicit.description == "CUSTOM-DESC"


# ---------------------------------------------------------------------------
# resolve_metric
# ---------------------------------------------------------------------------


async def test_resolve_metric_returns_judge_when_enabled() -> None:
    client = FakeLLMClient(script=[_judge_resp("5")])
    metric, mode = resolve_metric(_cfg(), client)
    assert mode == "judge"
    result = await metric(
        Example(inputs={"conversation": "user: hi"}),
        _prediction({"answer": "hello"}),
    )
    assert result.score == 1.0


def test_resolve_metric_exact_match_when_disabled() -> None:
    cfg = _cfg(judge={"enabled": False})
    metric, mode = resolve_metric(cfg, FakeLLMClient())
    assert mode == "exact-match"
    assert metric is default_metric


def test_resolve_metric_exact_match_when_no_models() -> None:
    cfg = _cfg(models={"task": "", "judge": ""})
    metric, mode = resolve_metric(cfg, FakeLLMClient())
    assert mode == "exact-match"
    assert metric is default_metric


# ---------------------------------------------------------------------------
# resolve_certificate_path
# ---------------------------------------------------------------------------


def test_certificate_path_defaults_to_registry_location() -> None:
    assert resolve_certificate_path(_cfg()) == Path(".reg/certificates/helpfulness.json")
    explicit = _cfg(judge={"certificate": "/tmp/cert.json"})
    assert resolve_certificate_path(explicit) == Path("/tmp/cert.json")
    other = _cfg(judge={"criterion": "correctness"})
    assert resolve_certificate_path(other) == Path(".reg/certificates/correctness.json")
