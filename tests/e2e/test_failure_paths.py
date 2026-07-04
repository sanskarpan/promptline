"""Failure-path end-to-end tests: the pipeline must fail loudly and safely.

Covers: uncalibrated judges blocking the gate, budget exhaustion mid-GEPA
(with checkpoint resume), null candidates being rejected without touching the
active pointer, contaminated/undersized gate splits, and a judge whose every
sample is unparseable scoring 0.0 instead of crashing the harness.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from promptline.core.types import Candidate, ModuleState
from promptline.eval.harness import Budget
from promptline.judge.calibrator import (
    CalibrationCertificate,
    UncalibratedJudgeError,
)
from promptline.judge.judge import PointwiseJudge, RubricCriterion
from promptline.optimizers.gepa import GEPA
from promptline.registry.gate import GateSettings, run_gate
from promptline.registry.registry import PromptRegistry
from tests.e2e.conftest import (
    MARKER,
    make_harness,
    make_judge_client,
    make_pipeline_client,
    marker_metric,
    seed_for,
    support_program,
    support_trainset,
)

# ---------------------------------------------------------------------------
# Uncalibrated judge blocks the gate
# ---------------------------------------------------------------------------


def _failed_certificate() -> CalibrationCertificate:
    return CalibrationCertificate(
        judge_name="pointwise:fake/judge",
        criterion="helpfulness",
        kappa=0.12,
        spearman=0.2,
        n_holdout=25,
        threshold=0.6,
        passed=False,
        judge_candidate_id="judge-1",
        created_at=datetime.now(UTC).isoformat(),
        confusion=[[0] * 5 for _ in range(5)],
        binning="identity",
    )


async def _gate_with_cert(tmp_path: Path, cert_path: Path):
    program = support_program()
    seed = seed_for(program)
    challenger = Candidate(
        id="challenger-1",
        modules={"support": ModuleState(instruction=f"Answer. {MARKER}.")},
    )
    settings = GateSettings(min_examples=5, require_certificate_path=cert_path)
    return await run_gate(
        program=program,
        incumbent=seed,
        candidates=[challenger],
        dev=support_trainset(10, "dev"),
        val=support_trainset(10, "val"),
        harness=make_harness(make_pipeline_client()),
        metric=marker_metric,
        settings=settings,
    )


async def test_gate_refuses_missing_certificate(tmp_path: Path) -> None:
    missing = tmp_path / "certificates" / "helpfulness.json"
    with pytest.raises(UncalibratedJudgeError, match="no calibration certificate"):
        await _gate_with_cert(tmp_path, missing)


async def test_gate_refuses_failed_certificate(tmp_path: Path) -> None:
    cert_path = tmp_path / "certificates" / "helpfulness.json"
    _failed_certificate().save(cert_path)
    with pytest.raises(UncalibratedJudgeError, match="not sufficient"):
        await _gate_with_cert(tmp_path, cert_path)


# ---------------------------------------------------------------------------
# Budget exhaustion mid-GEPA: best-so-far + resumable checkpoint
# ---------------------------------------------------------------------------


async def test_gepa_budget_wall_then_resume(tmp_path: Path) -> None:
    program = support_program()
    seed = seed_for(program)
    trainset = support_trainset(8, "gepa")
    run_dir = tmp_path / "run"

    # Budget dies mid-iteration: seed full-eval takes 4 rollouts (D_pareto=4),
    # parent minibatch 3 more; the child minibatch hits the wall at 9.
    small = Budget(max_rollouts=9)
    first = await GEPA(
        minibatch_size=3, max_iterations=10, use_merge=False, run_dir=run_dir
    ).optimize(
        program, seed, trainset, marker_metric,
        small, make_harness(make_pipeline_client()),
    )

    assert small.exhausted
    # Best-so-far is returned (only the seed was fully evaluated).
    assert first.best.id == seed.id
    assert (run_dir / "checkpoint.json").exists()
    assert (run_dir / "events.jsonl").exists()

    # Resume with a bigger budget: the pool grows past the checkpointed state
    # and the optimizer now finds the marker instruction.
    resumed = await GEPA(
        minibatch_size=3,
        max_iterations=10,
        use_merge=False,
        run_dir=run_dir,
        resume_from=run_dir,
    ).optimize(
        program, seed, trainset, marker_metric,
        Budget(max_rollouts=100), make_harness(make_pipeline_client()),
    )

    assert len(resumed.candidates) > len(first.candidates)
    assert resumed.best.id != seed.id
    assert MARKER in resumed.best.modules["support"].instruction
    assert resumed.scores[resumed.best.id] > resumed.scores[seed.id]


# ---------------------------------------------------------------------------
# Gate rejects null candidates; the active pointer never moves
# ---------------------------------------------------------------------------


async def test_gate_rejects_noise_candidates_no_activation(tmp_path: Path) -> None:
    program = support_program()
    incumbent = seed_for(program)
    # Five "noise" challengers: distinct instructions, none contains the
    # marker, so every one scores exactly like the incumbent (all deltas 0).
    noise = [
        Candidate(
            id=f"noise-{i}",
            modules={"support": ModuleState(instruction=f"Try harder, variant {i}.")},
        )
        for i in range(5)
    ]

    registry = PromptRegistry(tmp_path / "registry")
    registry.register(incumbent, "support")
    for cand in noise:
        registry.register(cand, "support")
    registry.activate("support", incumbent.id)

    report = await run_gate(
        program=program,
        incumbent=incumbent,
        candidates=noise,
        dev=support_trainset(60, "dev"),
        val=support_trainset(50, "val"),
        harness=make_harness(make_pipeline_client()),
        metric=marker_metric,
        settings=GateSettings(min_examples=50),
    )

    assert report.verdict == "reject"
    assert report.winner_id is None
    assert all(not r.holm_significant for r in report.results)

    # Promotion protocol: only a promote verdict moves the pointer.
    if report.verdict == "promote" and report.winner_id:  # pragma: no cover
        registry.activate("support", report.winner_id, report.model_dump_json())
    active = registry.get_active("support")
    assert active is not None and active[0] == incumbent.id


# ---------------------------------------------------------------------------
# Gate refusals: contamination and undersized dev
# ---------------------------------------------------------------------------


async def test_gate_refuses_contaminated_splits() -> None:
    program = support_program()
    seed = seed_for(program)
    challenger = Candidate(
        id="c-1", modules={"support": ModuleState(instruction=f"A. {MARKER}.")}
    )
    shared = support_trainset(60, "shared")
    with pytest.raises(ValueError, match="contamination"):
        await run_gate(
            program=program,
            incumbent=seed,
            candidates=[challenger],
            dev=shared,
            val=shared[:50],
            harness=make_harness(make_pipeline_client()),
            metric=marker_metric,
            settings=GateSettings(min_examples=50),
        )


async def test_gate_refuses_undersized_dev() -> None:
    program = support_program()
    seed = seed_for(program)
    challenger = Candidate(
        id="c-1", modules={"support": ModuleState(instruction=f"A. {MARKER}.")}
    )
    with pytest.raises(ValueError, match="dev set too small"):
        await run_gate(
            program=program,
            incumbent=seed,
            candidates=[challenger],
            dev=support_trainset(10, "dev"),
            val=support_trainset(50, "val"),
            harness=make_harness(make_pipeline_client()),
            metric=marker_metric,
            settings=GateSettings(min_examples=50),
        )


# ---------------------------------------------------------------------------
# Judge k-sampling: all samples unparseable -> metric returns 0.0, no crash
# ---------------------------------------------------------------------------


async def test_unparseable_judge_scores_zero_inside_harness() -> None:
    program = support_program()
    seed = seed_for(program)
    judge = PointwiseJudge(
        criterion=RubricCriterion(name="helpfulness", description="helpful?"),
        judge_model="fake/judge",
        samples=3,
    )
    # Task client answers fine; every judge sample yields "[[score]]: N/A".
    metric = judge.as_metric(make_judge_client("unparseable"))
    harness = make_harness(make_pipeline_client())

    report = await harness.evaluate(
        program, seed, support_trainset(4, "judge"), metric
    )

    assert report.n == 4
    assert all(r.score == 0.0 for r in report.per_example)
    assert all("judge error" in r.feedback for r in report.per_example)
    # The harness itself never marked these as hard failures.
    assert all(not r.failed for r in report.per_example)
