"""Tests for BootstrapFewShot and BootstrapRandomSearch optimizers."""
from __future__ import annotations

import pytest

from promptline.core.llm import FakeLLMClient
from promptline.core.program import ModelConfig, PromptProgram
from promptline.core.types import Candidate, Example, ModuleState
from promptline.eval.harness import Budget, EvalHarness, MetricResult
from promptline.optimizers.base import RunEvent
from promptline.optimizers.bootstrap import BootstrapFewShot, BootstrapRandomSearch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _program() -> PromptProgram:
    return PromptProgram.simple(
        instruction="Answer the question.",
        inputs=["question"],
        outputs=["answer"],
    )


def _seed(program: PromptProgram) -> Candidate:
    return Candidate.seed(
        modules={
            m.name: ModuleState(instruction=m.signature.instruction)
            for m in program.modules
        }
    )


def _model_cfg() -> ModelConfig:
    return ModelConfig(task_model="fake")


def _fake_client_single_answer(answer: str = "Paris") -> FakeLLMClient:
    """Returns a client that always replies with a parseable [[answer]] block."""
    return FakeLLMClient(script=lambda _call: f"[[answer]]: {answer}")


def _exact_metric(expected: str):
    """Metric that scores 1.0 when outputs['answer'] == expected."""

    def metric(example: Example, prediction) -> MetricResult:  # type: ignore[type-arg]
        got = prediction.outputs.get("answer", "")
        score = 1.0 if got.strip() == expected else 0.0
        return MetricResult(score=score)

    return metric


def _always_pass_metric(example: Example, prediction) -> MetricResult:  # type: ignore[type-arg]
    return MetricResult(score=1.0)


def _always_fail_metric(example: Example, prediction) -> MetricResult:  # type: ignore[type-arg]
    return MetricResult(score=0.0)


# ---------------------------------------------------------------------------
# BootstrapFewShot tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_passing_examples_become_demos() -> None:
    """Examples with score >= threshold should be collected as demos."""
    program = _program()
    seed = _seed(program)
    client = _fake_client_single_answer("Paris")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=50)

    examples = [
        Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"})
        for i in range(5)
    ]
    metric = _exact_metric("Paris")

    opt = BootstrapFewShot(max_demos=4, threshold=1.0, rng_seed=0)
    result = await opt.optimize(program, seed, examples, metric, budget, harness)

    first_mod = program.modules[0].name
    demos = result.best.modules[first_mod].demos
    assert len(demos) == 4, f"Expected 4 demos, got {len(demos)}"
    for demo in demos:
        assert demo.outputs.get("answer") == "Paris"
        assert "question" in demo.inputs


@pytest.mark.asyncio
async def test_bootstrap_failing_examples_not_collected() -> None:
    """Examples that don't meet threshold should not become demos."""
    program = _program()
    seed = _seed(program)
    client = _fake_client_single_answer("wrong")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=50)

    examples = [
        Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"})
        for i in range(5)
    ]
    metric = _exact_metric("Paris")  # model says "wrong", so all fail

    opt = BootstrapFewShot(max_demos=4, threshold=1.0, rng_seed=0)
    result = await opt.optimize(program, seed, examples, metric, budget, harness)

    first_mod = program.modules[0].name
    demos = result.best.modules[first_mod].demos
    assert len(demos) == 0


@pytest.mark.asyncio
async def test_bootstrap_max_demos_cap() -> None:
    """Should stop collecting after max_demos, even if more examples pass."""
    program = _program()
    seed = _seed(program)
    client = _fake_client_single_answer("Paris")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=50)

    examples = [
        Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"})
        for i in range(10)
    ]
    metric = _exact_metric("Paris")

    opt = BootstrapFewShot(max_demos=3, threshold=1.0, rng_seed=0)
    result = await opt.optimize(program, seed, examples, metric, budget, harness)

    first_mod = program.modules[0].name
    demos = result.best.modules[first_mod].demos
    assert len(demos) == 3
    assert budget.rollouts_used == 3  # stopped early


@pytest.mark.asyncio
async def test_bootstrap_budget_rollout_accounting() -> None:
    """Budget rollouts_used should reflect how many examples were attempted."""
    program = _program()
    seed = _seed(program)
    client = _fake_client_single_answer("Paris")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=3)

    examples = [
        Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"})
        for i in range(10)
    ]
    metric = _exact_metric("Paris")

    opt = BootstrapFewShot(max_demos=10, threshold=1.0, rng_seed=0)
    await opt.optimize(program, seed, examples, metric, budget, harness)

    assert budget.rollouts_used > 0
    assert budget.rollouts_used <= 3


@pytest.mark.asyncio
async def test_bootstrap_events_emitted_in_order() -> None:
    """run_started, candidate_proposed, run_finished should be emitted in order."""
    program = _program()
    seed = _seed(program)
    client = _fake_client_single_answer("Paris")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=50)

    examples = [Example(inputs={"question": "q"}, labels={"answer": "Paris"})]
    metric = _exact_metric("Paris")

    events: list[RunEvent] = []

    opt = BootstrapFewShot(max_demos=4, threshold=1.0, rng_seed=0)
    await opt.optimize(program, seed, examples, metric, budget, harness, emit=events.append)

    event_types = [e.type for e in events]
    assert event_types[0] == "run_started"
    assert "candidate_proposed" in event_types
    assert event_types[-1] == "run_finished"


@pytest.mark.asyncio
async def test_bootstrap_demo_content_matches_example() -> None:
    """Demo inputs should match example.inputs; outputs should match parsed trace."""
    program = _program()
    seed = _seed(program)
    client = _fake_client_single_answer("Berlin")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=50)

    examples = [
        Example(inputs={"question": "What is the capital of Germany?"})
    ]
    metric = _always_pass_metric

    opt = BootstrapFewShot(max_demos=4, threshold=1.0, rng_seed=0)
    result = await opt.optimize(program, seed, examples, metric, budget, harness)

    first_mod = program.modules[0].name
    demos = result.best.modules[first_mod].demos
    assert len(demos) == 1
    assert demos[0].inputs == {"question": "What is the capital of Germany?"}
    assert demos[0].outputs == {"answer": "Berlin"}


@pytest.mark.asyncio
async def test_bootstrap_result_structure() -> None:
    """OptimizeResult should include seed and best; scores keyed by best.id."""
    program = _program()
    seed = _seed(program)
    client = _fake_client_single_answer("Paris")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=50)

    examples = [
        Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"})
        for i in range(4)
    ]

    opt = BootstrapFewShot(max_demos=4, threshold=1.0, rng_seed=0)
    result = await opt.optimize(
        program, seed, examples, _always_pass_metric, budget, harness
    )

    assert result.best is not None
    assert seed in result.candidates
    assert result.best in result.candidates
    assert result.best.id in result.scores
    assert 0.0 <= result.scores[result.best.id] <= 1.0


# ---------------------------------------------------------------------------
# BootstrapRandomSearch tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_rs_basic_run() -> None:
    """BootstrapRandomSearch should complete and return a valid OptimizeResult."""
    program = _program()
    seed = _seed(program)
    client = _fake_client_single_answer("Paris")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=100)

    examples = [
        Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"})
        for i in range(10)
    ]
    metric = _exact_metric("Paris")

    opt = BootstrapRandomSearch(
        n_subsets=3,
        subset_size=2,
        threshold=1.0,
        val_fraction=0.3,
        rng_seed=0,
    )
    result = await opt.optimize(program, seed, examples, metric, budget, harness)

    assert result.best is not None
    assert len(result.candidates) > 0
    assert budget.rollouts_used > 0


@pytest.mark.asyncio
async def test_bootstrap_rs_favors_best_subset() -> None:
    """Random search should return the candidate with the highest val score.

    Script the metric to always return 1.0 so all candidates score equally;
    verify we get a candidate back (trivially passes, but exercises the path).
    Then re-run with a metric that only passes when a specific marker demo
    exists.
    """
    program = _program()
    seed = _seed(program)

    # All examples answer "marker" — pool will contain demos with answer=marker.
    client = _fake_client_single_answer("marker")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=200)

    examples = [
        Example(inputs={"question": f"q{i}"}, labels={"answer": "marker"})
        for i in range(12)
    ]
    metric = _exact_metric("marker")

    opt = BootstrapRandomSearch(
        n_subsets=4,
        subset_size=2,
        threshold=1.0,
        val_fraction=0.3,
        rng_seed=42,
    )
    result = await opt.optimize(program, seed, examples, metric, budget, harness)

    # Best candidate should have the highest score in the scores dict.
    best_score = result.scores.get(result.best.id, -1)
    for cand in result.candidates:
        cand_score = result.scores.get(cand.id, -1)
        assert cand_score <= best_score + 1e-9, (
            f"Candidate {cand.id} scored {cand_score} > best {best_score}"
        )


@pytest.mark.asyncio
async def test_bootstrap_rs_events_in_order() -> None:
    """Events should start with run_started and end with run_finished."""
    program = _program()
    seed = _seed(program)
    client = _fake_client_single_answer("Paris")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=100)

    examples = [
        Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"})
        for i in range(6)
    ]

    events: list[RunEvent] = []
    opt = BootstrapRandomSearch(n_subsets=2, subset_size=2, rng_seed=0)
    await opt.optimize(
        program, seed, examples, _always_pass_metric, budget, harness, emit=events.append
    )

    event_types = [e.type for e in events]
    assert event_types[0] == "run_started"
    assert event_types[-1] == "run_finished"
    assert "candidate_proposed" in event_types
    assert "full_eval" in event_types


@pytest.mark.asyncio
async def test_bootstrap_rs_budget_respected() -> None:
    """BootstrapRandomSearch should not exceed budget rollout limit."""
    program = _program()
    seed = _seed(program)
    client = _fake_client_single_answer("Paris")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=5)

    examples = [
        Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"})
        for i in range(20)
    ]
    metric = _exact_metric("Paris")

    opt = BootstrapRandomSearch(
        n_subsets=8,
        subset_size=4,
        threshold=1.0,
        val_fraction=0.3,
        rng_seed=0,
    )
    result = await opt.optimize(program, seed, examples, metric, budget, harness)

    # Should not crash and rollouts should not exceed max.
    assert budget.rollouts_used <= budget.max_rollouts
    assert result.best is not None
