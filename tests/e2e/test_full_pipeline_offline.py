"""Offline end-to-end: calibrate -> optimize -> gate -> registry -> serve.

Real components throughout; only the LLM is a scripted FakeLLMClient.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from promptline.eval.harness import Budget
from promptline.judge.calibrator import Calibrator
from promptline.judge.judge import PointwiseJudge, RubricCriterion
from promptline.optimizers.base import RunEvent
from promptline.optimizers.bootstrap import BootstrapFewShot, BootstrapRandomSearch
from promptline.optimizers.gepa import GEPA
from promptline.optimizers.mipro import MIPRO
from promptline.optimizers.opro import OPRO
from promptline.optimizers.protegi import ProTeGi
from promptline.registry.gate import GateSettings, run_gate
from promptline.registry.registry import PromptRegistry
from promptline.server.app import create_app
from promptline.server.runs import RunManager
from tests.e2e.conftest import (
    MARKER,
    build_gold_dataset,
    echo_metric,
    make_echo_client,
    make_harness,
    make_judge_client,
    make_pipeline_client,
    marker_metric,
    seed_for,
    support_program,
    support_trainset,
)

CRITERION = RubricCriterion(
    name="helpfulness",
    description="How well the response addresses the customer's request.",
)


# ---------------------------------------------------------------------------
# THE headline test: the full chain in one pass
# ---------------------------------------------------------------------------


async def test_full_pipeline_offline(tmp_path: Path) -> None:
    # ---- (1) Calibrate the judge against gold human labels ----------------
    judge = PointwiseJudge(criterion=CRITERION, judge_model="fake/judge")
    gold = build_gold_dataset(30)
    calibrator = Calibrator(judge, gold, make_judge_client("high"))
    cert = await calibrator.calibrate()

    assert cert.passed is True
    assert cert.kappa >= 0.6
    assert cert.n_holdout == len([r for r in calibrator.holdout if r.reference_output is not None])
    cert_path = tmp_path / "certificates" / "helpfulness.json"
    cert.save(cert_path)

    # ---- (2) GEPA-optimize the mediocre seed prompt ------------------------
    program = support_program()
    seed = seed_for(program)
    run_dir = tmp_path / "runs" / "gepa-e2e"
    client = make_pipeline_client()
    optimizer = GEPA(minibatch_size=3, max_iterations=4, use_merge=False, run_dir=run_dir)
    result = await optimizer.optimize(
        program=program,
        seed=seed,
        trainset=support_trainset(20, "train"),
        metric=marker_metric,
        budget=Budget(max_rollouts=200),
        harness=make_harness(client),
    )
    best = result.best

    assert best.id != seed.id
    assert result.scores[best.id] > result.scores[seed.id]
    assert MARKER in best.modules["support"].instruction
    # Lineage depth >= 2: the winner records its parent chain back to the seed.
    assert best.parent_ids, "winner must have recorded lineage"
    assert (run_dir / "events.jsonl").exists()

    # ---- (3) Statistical gate: best challenges the seed --------------------
    settings = GateSettings(min_examples=50, require_certificate_path=cert_path, min_kappa=0.6)
    report = await run_gate(
        program=program,
        incumbent=seed,
        candidates=[best],
        dev=support_trainset(60, "dev"),
        val=support_trainset(50, "val"),
        harness=make_harness(client),
        metric=marker_metric,
        settings=settings,
    )

    assert report.verdict == "promote"
    assert report.winner_id == best.id
    assert report.results[0].holm_significant is True
    assert report.val_ci_low is not None and report.val_ci_low > 0

    # ---- (4) Registry: register both, activate the gated winner ------------
    registry = PromptRegistry(tmp_path / "registry")
    registry.register(seed, "support")
    registry.register(best, "support", run_id="gepa-e2e")
    registry.activate("support", seed.id)  # baseline bootstrap
    registry.activate("support", best.id, report.model_dump_json())

    active = registry.get_active("support")
    assert active is not None and active[0] == best.id
    assert seed.id in registry.lineage(best.id)

    # ---- (5) Serving plane: active prompt + ETag revalidation --------------
    app = create_app(registry, RunManager(tmp_path / "server-runs"))
    with TestClient(app) as http:
        resp = http.get("/prompts/support/active")
        assert resp.status_code == 200
        body = resp.json()
        assert body["prompt_id"] == best.id
        assert MARKER in body["modules"]["support"]["instruction"]
        etag = resp.headers["ETag"]
        assert etag == f'"{best.id}"'

        cached = http.get("/prompts/support/active", headers={"If-None-Match": etag})
        assert cached.status_code == 304


# ---------------------------------------------------------------------------
# Parametrized mini-chain: every other optimizer produces a sane result
# ---------------------------------------------------------------------------


def _mini_setups():
    return {
        "bootstrap": (
            BootstrapFewShot(max_demos=2),
            make_echo_client,
            echo_metric,
            6,
            40,
        ),
        "bootstrap-rs": (
            BootstrapRandomSearch(n_subsets=2, subset_size=2),
            make_echo_client,
            echo_metric,
            6,
            60,
        ),
        "opro": (
            OPRO(n_steps=2, candidates_per_step=2, minibatch_size=4),
            make_pipeline_client,
            marker_metric,
            8,
            60,
        ),
        "protegi": (
            ProTeGi(
                beam_width=2,
                n_gradients=1,
                n_paraphrases=1,
                n_rounds=1,
                minibatch_size=4,
                racing_rounds=1,
                racing_batch=4,
            ),
            make_pipeline_client,
            marker_metric,
            8,
            80,
        ),
        "mipro": (
            MIPRO(
                n_instruction_candidates=2,
                n_demo_sets=2,
                demos_per_set=2,
                n_trials=4,
                minibatch_size=4,
                full_eval_steps=2,
            ),
            make_pipeline_client,
            marker_metric,
            8,
            120,
        ),
    }


@pytest.mark.parametrize("name", list(_mini_setups().keys()))
async def test_optimizer_mini_chain(name: str) -> None:
    optimizer, client_factory, metric, n_train, max_rollouts = _mini_setups()[name]
    program = support_program()
    seed = seed_for(program)
    client = client_factory()
    budget = Budget(max_rollouts=max_rollouts)

    events: list[RunEvent] = []
    result = await optimizer.optimize(
        program=program,
        seed=seed,
        trainset=support_trainset(n_train, f"mini-{name}"),
        metric=metric,
        budget=budget,
        harness=make_harness(client),
        emit=events.append,
    )

    # Best candidate exists and its modules are intact.
    assert result.best is not None
    assert set(result.best.modules) == {"support"}
    # Scores populated with finite values.
    assert result.scores
    assert all(s == s for s in result.scores.values())
    # Events were emitted, bracketed by run_started/run_finished.
    assert events, "optimizer must emit run events"
    assert events[0].type == "run_started"
    assert events[-1].type == "run_finished"
    assert result.events_count == len(events)
    # Budget respected.
    assert budget.rollouts_used <= max_rollouts

    if metric is marker_metric:
        # Proposal-driven optimizers must have found the marker instruction.
        assert MARKER in result.best.modules["support"].instruction
        assert result.scores[result.best.id] == 1.0
