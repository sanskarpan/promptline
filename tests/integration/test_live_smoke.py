"""Opt-in live smoke test against real OpenRouter models.

Run with an API key: ``OPENROUTER_API_KEY=sk-or-... make live-smoke``
(or ``uv run pytest -m live tests/integration``).  Without the key the whole
module is skipped, so ``uv run pytest`` stays offline and free.

Budget-capped hard at $0.50 / 20 rollouts: a tiny GEPA run over 5 support
examples with a judge-as-metric on claude-3.5-haiku, then a deliberately
undersized (min_examples=5) gate run that only has to *produce a report*.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from promptline.core.program import ModelConfig, PromptProgram
from promptline.core.types import Candidate, Example, ModuleState
from promptline.eval.harness import Budget, EvalHarness
from promptline.judge.judge import PointwiseJudge, RubricCriterion
from promptline.optimizers.gepa import GEPA
from promptline.registry.gate import GateSettings, run_gate

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="live smoke needs OPENROUTER_API_KEY",
    ),
]

TASK_MODEL = "meta-llama/llama-3.1-8b-instruct"
JUDGE_MODEL = "anthropic/claude-3.5-haiku"

SUPPORT_QUESTIONS = [
    "How do I reset my account password?",
    "My last invoice was charged twice — how do I get a refund?",
    "How can I change the shipping address on an open order?",
    "The mobile app crashes when I open settings. What should I do?",
    "How do I cancel my subscription before the next billing cycle?",
]


def _examples(prefix: str, questions: list[str]) -> list[Example]:
    return [
        Example(inputs={"conversation": f"user: [{prefix}-{i}] {q}"})
        for i, q in enumerate(questions)
    ]


async def test_live_gepa_then_gate_smoke(tmp_path: Path) -> None:
    from promptline.core.openrouter import OpenRouterClient

    client = OpenRouterClient()
    program = PromptProgram.simple(
        instruction="You are a support agent. Answer the question.",
        inputs=["conversation"],
        outputs=["answer"],
        name="support",
    )
    seed = Candidate.seed(
        {"support": ModuleState(instruction=program.modules[0].signature.instruction)}
    )

    judge = PointwiseJudge(
        criterion=RubricCriterion(
            name="helpfulness",
            description=(
                "How well the response addresses the customer's request with "
                "actionable, relevant information."
            ),
        ),
        judge_model=JUDGE_MODEL,
    )
    metric = judge.as_metric(client)
    harness = EvalHarness(
        client,
        ModelConfig(
            task_model=TASK_MODEL,
            reflection_model=JUDGE_MODEL,
            judge_model=JUDGE_MODEL,
        ),
        concurrency=2,
    )

    # ---- GEPA: 5 examples, hard caps ---------------------------------------
    budget = Budget(max_rollouts=20, max_cost_usd=0.50)
    result = await GEPA(
        minibatch_size=2,
        n_pareto=2,
        max_iterations=4,
        use_merge=False,
        run_dir=tmp_path / "run",
    ).optimize(
        program=program,
        seed=seed,
        trainset=_examples("train", SUPPORT_QUESTIONS),
        metric=metric,
        budget=budget,
        harness=harness,
    )

    assert result.best is not None
    assert result.scores, "live run must record scores"
    assert budget.rollouts_used <= 20
    assert 0.0 < budget.cost_used < 0.50, (
        f"cost must be recorded and under the cap, got {budget.cost_used}"
    )
    assert (tmp_path / "run" / "events.jsonl").exists()

    # ---- Gate: 5 dev / 5 val, min_examples=5 -> any verdict is fine ---------
    gate_budget = Budget(max_cost_usd=0.50)
    report = await run_gate(
        program=program,
        incumbent=seed,
        candidates=[result.best]
        if result.best.id != seed.id
        else [
            Candidate.seed({"support": ModuleState(instruction="Answer politely and cite steps.")})
        ],
        dev=_examples("dev", SUPPORT_QUESTIONS),
        val=_examples("val", SUPPORT_QUESTIONS),
        harness=harness,
        metric=metric,
        settings=GateSettings(min_examples=5),
        budget=gate_budget,
    )

    assert report.verdict in ("promote", "reject")
    assert len(report.results) == 1
    assert gate_budget.cost_used < 0.50
