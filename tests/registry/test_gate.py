from __future__ import annotations

from pathlib import Path

import pytest

from promptline.core.llm import FakeLLMClient, LLMCall
from promptline.core.program import ModelConfig, Prediction, PromptProgram
from promptline.core.types import Candidate, Example, ModuleState
from promptline.eval.harness import EvalHarness, MetricResult
from promptline.judge.calibrator import UncalibratedJudgeError
from promptline.registry.gate import (
    CandidateGateResult,
    GateReport,
    GateSettings,
    run_gate,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
#
# The fake client echoes each candidate's instruction back as the answer, and
# the scripted metric assigns deterministic per-example scores based on the
# answer token and the example index:
#   base     -> 0.5 everywhere (incumbent)
#   good*    -> 0.8 everywhere (true winner, uniform +0.3 delta)
#   noise    -> 0.5 +/- 0.1 alternating (mean delta 0)
#   null*    -> 0.5 +/- 0.01 alternating (tiny, non-significant deltas)
# ---------------------------------------------------------------------------


def _echo_script(call: LLMCall) -> str:
    instruction = call.messages[0].content.splitlines()[0]
    return f"[[answer]]: {instruction}"


def _program() -> PromptProgram:
    return PromptProgram.simple(
        instruction="base", inputs=["question"], outputs=["answer"]
    )


def _cand(instruction: str) -> Candidate:
    return Candidate(
        id=f"cand-{instruction[:16]}",
        modules={"main": ModuleState(instruction=instruction)},
    )


def _examples(n: int, offset: int = 0) -> list[Example]:
    return [Example(inputs={"question": str(offset + i)}) for i in range(n)]


def _metric(example: Example, prediction: Prediction) -> MetricResult:
    answer = prediction.outputs.get("answer", "")
    idx = int(example.inputs["question"])
    if answer.startswith("good"):
        return MetricResult(score=0.8)
    if answer.startswith("noise"):
        return MetricResult(score=0.5 + (0.1 if idx % 2 == 0 else -0.1))
    if answer.startswith("null"):
        return MetricResult(score=0.5 + (0.01 if idx % 2 == 0 else -0.01))
    return MetricResult(score=0.5)


def _harness() -> EvalHarness:
    client = FakeLLMClient(script=_echo_script)
    return EvalHarness(client, ModelConfig(task_model="test-model"), concurrency=8)


def _settings(**overrides) -> GateSettings:
    return GateSettings(**overrides)


DEV = _examples(60)
VAL = _examples(50, offset=1000)


# ---------------------------------------------------------------------------
# Promotion / rejection
# ---------------------------------------------------------------------------


async def test_clear_winner_promoted() -> None:
    winner = _cand("good")
    report = await run_gate(
        program=_program(),
        incumbent=_cand("base"),
        candidates=[winner],
        dev=DEV,
        val=VAL,
        harness=_harness(),
        metric=_metric,
        settings=_settings(),
    )
    assert report.verdict == "promote"
    assert report.winner_id == winner.id
    assert len(report.results) == 1
    result = report.results[0]
    assert result.holm_significant is True
    assert result.mean_delta == pytest.approx(0.3)
    assert result.dev_mean == pytest.approx(0.8)
    assert result.incumbent_dev_mean == pytest.approx(0.5)
    assert report.val_mean_delta == pytest.approx(0.3)
    assert report.val_ci_low is not None and report.val_ci_low > 0
    assert "verbosity" not in report.flags
    assert len(report.spot_samples) == 5
    assert report.spot_samples[0]["example_idx"] == 0
    assert report.spot_samples[0]["output"] == "good"
    assert report.spot_samples[0]["score"] == pytest.approx(0.8)


async def test_noise_only_candidate_rejected() -> None:
    report = await run_gate(
        program=_program(),
        incumbent=_cand("base"),
        candidates=[_cand("noise")],
        dev=DEV,
        val=VAL,
        harness=_harness(),
        metric=_metric,
        settings=_settings(),
    )
    assert report.verdict == "reject"
    assert report.winner_id is None
    assert report.results[0].holm_significant is False
    assert report.val_mean_delta is None


async def test_holm_only_true_winner_survives() -> None:
    winner = _cand("good")
    nulls = [_cand(f"null{k}") for k in range(5)]
    report = await run_gate(
        program=_program(),
        incumbent=_cand("base"),
        candidates=[*nulls, winner],
        dev=DEV,
        val=VAL,
        harness=_harness(),
        metric=_metric,
        settings=_settings(),
    )
    assert report.verdict == "promote"
    assert report.winner_id == winner.id
    significant = [r.candidate_id for r in report.results if r.holm_significant]
    assert significant == [winner.id]


async def test_all_nulls_rejected() -> None:
    report = await run_gate(
        program=_program(),
        incumbent=_cand("base"),
        candidates=[_cand(f"null{k}") for k in range(6)],
        dev=DEV,
        val=VAL,
        harness=_harness(),
        metric=_metric,
        settings=_settings(),
    )
    assert report.verdict == "reject"
    assert report.winner_id is None
    assert not any(r.holm_significant for r in report.results)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


async def test_undersized_dev_raises() -> None:
    with pytest.raises(ValueError, match="examples"):
        await run_gate(
            program=_program(),
            incumbent=_cand("base"),
            candidates=[_cand("good")],
            dev=_examples(10),
            val=VAL,
            harness=_harness(),
            metric=_metric,
            settings=_settings(),
        )


async def test_dev_val_contamination_raises() -> None:
    with pytest.raises(ValueError, match="contamin"):
        await run_gate(
            program=_program(),
            incumbent=_cand("base"),
            candidates=[_cand("good")],
            dev=DEV,
            val=[*VAL, DEV[5]],
            harness=_harness(),
            metric=_metric,
            settings=_settings(),
        )


async def test_empty_candidates_raises() -> None:
    with pytest.raises(ValueError, match="candidate"):
        await run_gate(
            program=_program(),
            incumbent=_cand("base"),
            candidates=[],
            dev=DEV,
            val=VAL,
            harness=_harness(),
            metric=_metric,
            settings=_settings(),
        )


async def test_missing_certificate_raises(tmp_path: Path) -> None:
    settings = _settings(require_certificate_path=tmp_path / "cert.json")
    with pytest.raises(UncalibratedJudgeError):
        await run_gate(
            program=_program(),
            incumbent=_cand("base"),
            candidates=[_cand("good")],
            dev=DEV,
            val=VAL,
            harness=_harness(),
            metric=_metric,
            settings=settings,
        )


# ---------------------------------------------------------------------------
# Tripwires
# ---------------------------------------------------------------------------


async def test_verbosity_flag_does_not_block_promotion() -> None:
    verbose_winner = _cand("good" + "x" * 300)
    report = await run_gate(
        program=_program(),
        incumbent=_cand("base"),
        candidates=[verbose_winner],
        dev=DEV,
        val=VAL,
        harness=_harness(),
        metric=_metric,
        settings=_settings(),
    )
    assert "verbosity" in report.flags
    assert report.verdict == "promote"
    assert report.winner_id == verbose_winner.id


async def test_small_val_warning() -> None:
    report = await run_gate(
        program=_program(),
        incumbent=_cand("base"),
        candidates=[_cand("good")],
        dev=DEV,
        val=_examples(20, offset=1000),
        harness=_harness(),
        metric=_metric,
        settings=_settings(),
    )
    assert any("20" in w for w in report.warnings)


async def test_val_truncation_warning_but_still_promotes() -> None:
    """Paired val n below min_examples warns but does not block (n >= 10)."""
    report = await run_gate(
        program=_program(),
        incumbent=_cand("base"),
        candidates=[_cand("good")],
        dev=DEV,
        val=_examples(20, offset=1000),
        harness=_harness(),
        metric=_metric,
        settings=_settings(),
    )
    assert any(w.startswith("val truncated to n=20") for w in report.warnings)
    assert report.verdict == "promote"
    assert "val_too_small" not in report.flags


async def test_tiny_paired_val_forces_reject() -> None:
    """Paired val n < 10 must force the verdict to reject with a flag."""
    report = await run_gate(
        program=_program(),
        incumbent=_cand("base"),
        candidates=[_cand("good")],
        dev=DEV,
        val=_examples(8, offset=1000),
        harness=_harness(),
        metric=_metric,
        settings=_settings(),
    )
    assert report.verdict == "reject"
    assert "val_too_small" in report.flags
    assert any(w.startswith("val truncated to n=8") for w in report.warnings)


# ---------------------------------------------------------------------------
# Settings plumbing
# ---------------------------------------------------------------------------


def test_settings_from_config_plumbs_certificate_and_kappa() -> None:
    from promptline.core.config import GateConfig

    cfg = GateConfig(
        alpha=0.01,
        min_examples=30,
        certificate="certs/helpfulness.json",
        min_kappa=0.7,
    )
    settings = GateSettings.from_config(cfg)
    assert settings.alpha == 0.01
    assert settings.min_examples == 30
    assert settings.require_certificate_path == Path("certs/helpfulness.json")
    assert settings.min_kappa == 0.7

    default = GateSettings.from_config(GateConfig())
    assert default.require_certificate_path is None


# ---------------------------------------------------------------------------
# Family-wise error rate under the null (seed sweep)
# ---------------------------------------------------------------------------


async def test_seed_sweep_null_candidates_fwer_controlled() -> None:
    """5 noisy null candidates over 10 seeds should almost never promote.

    Per-example scores of each null candidate are the incumbent's (0.5)
    plus per-seed N(0, 0.1) noise, so every true delta is zero.  With
    alpha=0.05 and Holm correction we expect ~0-1 promotions across 10
    seeds; the bound of 2 is deliberately loose to avoid flakiness while
    still demonstrating family-wise error control.
    """
    import random as _random

    dev = _examples(60)
    val = _examples(50, offset=1000)
    promotions = 0
    for seed in range(10):
        rng = _random.Random(seed)
        # noise[k][idx] — per-candidate, per-example noise around 0.5.
        noise = [
            [rng.gauss(0.0, 0.1) for _ in range(1100)] for _ in range(5)
        ]

        def metric(example: Example, prediction: Prediction) -> MetricResult:
            answer = prediction.outputs.get("answer", "")
            idx = int(example.inputs["question"])
            if answer.startswith("fwer"):
                k = int(answer[4])
                return MetricResult(score=0.5 + noise[k][idx])  # noqa: B023
            return MetricResult(score=0.5)

        report = await run_gate(
            program=_program(),
            incumbent=_cand("base"),
            candidates=[_cand(f"fwer{k}") for k in range(5)],
            dev=dev,
            val=val,
            harness=_harness(),
            metric=metric,
            settings=_settings(),
        )
        if report.verdict == "promote":
            promotions += 1
    assert promotions <= 2, f"FWER not controlled: {promotions}/10 promotions"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


async def test_gate_report_json_round_trip() -> None:
    report = await run_gate(
        program=_program(),
        incumbent=_cand("base"),
        candidates=[_cand("good"), _cand("noise")],
        dev=DEV,
        val=VAL,
        harness=_harness(),
        metric=_metric,
        settings=_settings(),
    )
    restored = GateReport.model_validate_json(report.model_dump_json())
    assert restored == report


def test_candidate_gate_result_fields() -> None:
    result = CandidateGateResult(
        candidate_id="c1",
        mean_delta=0.3,
        ci_low=0.2,
        ci_high=0.4,
        p_value=0.001,
        holm_significant=True,
        dev_mean=0.8,
        incumbent_dev_mean=0.5,
    )
    assert result.candidate_id == "c1"
    assert result.holm_significant is True


def test_settings_from_config_prefers_gate_cert_falls_back_to_judge() -> None:
    """gate.certificate wins when set; judge.certificate is the fallback."""
    from promptline.core.config import GateConfig, JudgeConfig

    judge = JudgeConfig(certificate="judge-cert.json", min_kappa=0.7)

    # gate.certificate empty → judge.certificate + judge.min_kappa used.
    settings = GateSettings.from_config(GateConfig(), judge)
    assert settings.require_certificate_path == Path("judge-cert.json")
    assert settings.min_kappa == 0.7

    # gate.certificate set → back-compat: gate values win.
    gate_cfg = GateConfig(certificate="gate-cert.json", min_kappa=0.65)
    settings = GateSettings.from_config(gate_cfg, judge)
    assert settings.require_certificate_path == Path("gate-cert.json")
    assert settings.min_kappa == 0.65

    # Neither set → no certificate requirement from config.
    assert GateSettings.from_config(GateConfig(), JudgeConfig()).require_certificate_path is None
