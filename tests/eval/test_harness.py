from __future__ import annotations

import asyncio

import pytest

from promptline.core.llm import FakeLLMClient, LLMError, LLMResponse
from promptline.core.program import ModelConfig, Module, PromptProgram
from promptline.core.types import Candidate, Example, Field, ModuleState, Signature
from promptline.eval.harness import (
    Budget,
    BudgetExhausted,
    EvalHarness,
    EvalReport,
    ExampleResult,
    MetricResult,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _program() -> PromptProgram:
    """Single-output program: question → answer."""
    return PromptProgram.simple(
        instruction="Answer the question.",
        inputs=["question"],
        outputs=["answer"],
    )


def _two_output_program() -> PromptProgram:
    """Two-output program: plain text fails parse → repair path → Prediction.failure."""
    sig = Signature(
        instruction="Do it.",
        inputs=[Field("question")],
        outputs=[Field("a"), Field("b")],
    )
    return PromptProgram(modules=[Module(name="main", signature=sig)])


def _candidate(program: PromptProgram | None = None) -> Candidate:
    if program is None:
        return Candidate.seed(modules={"main": ModuleState(instruction="Answer the question.")})
    return Candidate.seed(
        modules={m.name: ModuleState(instruction=m.signature.instruction) for m in program.modules}
    )


def _cfg() -> ModelConfig:
    return ModelConfig(task_model="test-model")


def _examples(n: int) -> list[Example]:
    return [Example(inputs={"question": f"Q{i}"}) for i in range(n)]


def _fixed_metric(score: float):
    def metric(example: Example, prediction) -> MetricResult:
        return MetricResult(score=score, feedback="ok")

    return metric


# ---------------------------------------------------------------------------
# Test-only LLM client helpers
# ---------------------------------------------------------------------------


class _CostlyFakeLLMClient:
    """Always returns cost_usd=cost_per_call so cost-budget tests work."""

    def __init__(self, cost_per_call: float) -> None:
        self._cost = cost_per_call

    async def complete(self, call: object) -> LLMResponse:
        return LLMResponse(text="[[answer]]: ok", cost_usd=self._cost)


class _FailOnNthCallClient:
    """Raises LLMError on the Nth complete() call; succeeds otherwise."""

    def __init__(self, fail_on_call: int) -> None:
        self._fail_on = fail_on_call
        self._count = 0

    async def complete(self, call: object) -> LLMResponse:
        self._count += 1
        if self._count == self._fail_on:
            raise LLMError(f"forced failure on call {self._count}")
        return LLMResponse(text="[[answer]]: ok")


# ---------------------------------------------------------------------------
# MetricResult
# ---------------------------------------------------------------------------


def test_metric_result_defaults() -> None:
    mr = MetricResult(score=0.42)
    assert mr.feedback == ""
    assert mr.per_module == {}


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_budget_not_exhausted_initially() -> None:
    b = Budget(max_rollouts=3)
    assert not b.exhausted


def test_budget_exhausted_at_rollout_limit() -> None:
    b = Budget(max_rollouts=3)
    b.charge(rollouts=3)
    assert b.exhausted


def test_budget_not_yet_exhausted_below_limit() -> None:
    b = Budget(max_rollouts=3)
    b.charge(rollouts=2)
    assert not b.exhausted


def test_budget_exhausted_at_cost_limit() -> None:
    b = Budget(max_cost_usd=1.0)
    b.charge(cost=1.01)
    assert b.exhausted


def test_budget_remaining_rollouts() -> None:
    b = Budget(max_rollouts=5)
    b.charge(rollouts=2)
    assert b.remaining_rollouts == 3


def test_budget_remaining_rollouts_none_when_unlimited() -> None:
    b = Budget()
    assert b.remaining_rollouts is None


def test_budget_remaining_rollouts_clamps_to_zero() -> None:
    b = Budget(max_rollouts=2)
    b.charge(rollouts=5)
    assert b.remaining_rollouts == 0


# ---------------------------------------------------------------------------
# EvalReport
# ---------------------------------------------------------------------------


def test_eval_report_mean_score() -> None:
    r = EvalReport(
        per_example=[
            ExampleResult(example_idx=0, score=0.8, feedback="", cost_usd=0.0, failed=False),
            ExampleResult(example_idx=1, score=0.6, feedback="", cost_usd=0.0, failed=False),
        ]
    )
    assert r.mean_score == pytest.approx(0.7)


def test_eval_report_mean_score_empty() -> None:
    r = EvalReport(per_example=[])
    assert r.mean_score == 0.0


def test_eval_report_total_cost() -> None:
    r = EvalReport(
        per_example=[
            ExampleResult(example_idx=0, score=0.5, feedback="", cost_usd=0.01, failed=False),
            ExampleResult(example_idx=1, score=0.5, feedback="", cost_usd=0.02, failed=False),
        ]
    )
    assert r.total_cost == pytest.approx(0.03)


def test_eval_report_n() -> None:
    r = EvalReport(
        per_example=[
            ExampleResult(example_idx=0, score=0.5, feedback="", cost_usd=0.0, failed=False),
        ]
    )
    assert r.n == 1


# ---------------------------------------------------------------------------
# EvalHarness.evaluate — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_mean_score_all_same() -> None:
    fake = FakeLLMClient(script=["[[answer]]: ok" for _ in range(4)])
    harness = EvalHarness(fake, _cfg())
    report = await harness.evaluate(_program(), _candidate(), _examples(4), _fixed_metric(0.5))
    assert report.n == 4
    assert report.mean_score == pytest.approx(0.5)
    assert report.truncated is False


@pytest.mark.asyncio
async def test_evaluate_mean_score_scripted() -> None:
    """Different scores per example are averaged correctly."""
    fake = FakeLLMClient(script=["[[answer]]: ok" for _ in range(3)])
    scores_iter = iter([1.0, 0.5, 0.0])

    def varying_metric(example, prediction) -> MetricResult:
        return MetricResult(score=next(scores_iter))

    harness = EvalHarness(fake, _cfg(), concurrency=1)
    report = await harness.evaluate(_program(), _candidate(), _examples(3), varying_metric)
    assert report.mean_score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Budget truncation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_max_rollouts_truncates() -> None:
    fake = FakeLLMClient(script=["[[answer]]: ok" for _ in range(10)])
    harness = EvalHarness(fake, _cfg(), concurrency=1)
    budget = Budget(max_rollouts=3)
    report = await harness.evaluate(
        _program(), _candidate(), _examples(10), _fixed_metric(1.0), budget=budget
    )
    assert report.n == 3
    assert report.truncated is True


@pytest.mark.asyncio
async def test_budget_no_truncation_when_sufficient() -> None:
    fake = FakeLLMClient(script=["[[answer]]: ok" for _ in range(5)])
    harness = EvalHarness(fake, _cfg())
    budget = Budget(max_rollouts=10)
    report = await harness.evaluate(
        _program(), _candidate(), _examples(5), _fixed_metric(1.0), budget=budget
    )
    assert report.n == 5
    assert report.truncated is False


# ---------------------------------------------------------------------------
# Failed prediction: score 0, metric NOT called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_prediction_scores_zero_metric_not_called() -> None:
    metric_calls = 0

    def metric(example: Example, prediction) -> MetricResult:
        nonlocal metric_calls
        metric_calls += 1
        return MetricResult(score=1.0)

    # Two-output program + two bad responses → Prediction.failure after repair
    prog = _two_output_program()
    cand = Candidate.seed(modules={"main": ModuleState(instruction="Do it.")})
    # Need 2 bad responses: initial call + repair attempt
    fake = FakeLLMClient(script=["bad output", "still bad"])
    harness = EvalHarness(fake, _cfg())
    report = await harness.evaluate(prog, cand, _examples(1), metric)

    assert metric_calls == 0
    assert len(report.per_example) == 1
    result = report.per_example[0]
    assert result.score == 0.0
    assert result.failed is True
    assert "unparseable" in result.feedback


# ---------------------------------------------------------------------------
# Async and sync metrics both work
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_metric_is_awaited() -> None:
    async def async_metric(example: Example, prediction) -> MetricResult:
        await asyncio.sleep(0)
        return MetricResult(score=0.75, feedback="async-ok")

    fake = FakeLLMClient(script=["[[answer]]: ok"])
    harness = EvalHarness(fake, _cfg())
    report = await harness.evaluate(_program(), _candidate(), _examples(1), async_metric)
    assert report.mean_score == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_sync_metric_works() -> None:
    def sync_metric(example: Example, prediction) -> MetricResult:
        return MetricResult(score=0.9)

    fake = FakeLLMClient(script=["[[answer]]: ok"])
    harness = EvalHarness(fake, _cfg())
    report = await harness.evaluate(_program(), _candidate(), _examples(1), sync_metric)
    assert report.mean_score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Concurrency cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_limit_respected() -> None:
    """At most `concurrency` examples should be inside program.run+metric at once."""
    max_in_flight = 0
    current_in_flight = 0

    async def slow_metric(example: Example, prediction) -> MetricResult:
        nonlocal max_in_flight, current_in_flight
        current_in_flight += 1
        max_in_flight = max(max_in_flight, current_in_flight)
        await asyncio.sleep(0)  # yield so the event loop can schedule others
        current_in_flight -= 1
        return MetricResult(score=1.0)

    fake = FakeLLMClient(script=["[[answer]]: ok" for _ in range(6)])
    harness = EvalHarness(fake, _cfg(), concurrency=2)
    await harness.evaluate(_program(), _candidate(), _examples(6), slow_metric)

    assert max_in_flight <= 2


# ---------------------------------------------------------------------------
# Ordering: results ordered by example_idx regardless of completion order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_results_ordered_by_example_idx() -> None:
    fake = FakeLLMClient(script=["[[answer]]: ok" for _ in range(5)])
    harness = EvalHarness(fake, _cfg(), concurrency=5)
    report = await harness.evaluate(_program(), _candidate(), _examples(5), _fixed_metric(1.0))
    idxs = [r.example_idx for r in report.per_example]
    assert idxs == sorted(idxs)
    assert idxs == list(range(5))


# ---------------------------------------------------------------------------
# Finding 1: cost budget wall (TDD — these tests drive the fix)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cost_budget_wall_truncates() -> None:
    """max_cost_usd=$3 with 20 examples costing $1 each must truncate well before 20."""
    budget = Budget(max_cost_usd=3.0)
    concurrency = 2
    cost_per_example = 1.0

    client = _CostlyFakeLLMClient(cost_per_call=cost_per_example)
    harness = EvalHarness(client, _cfg(), concurrency=concurrency)
    report = await harness.evaluate(
        _program(), _candidate(), _examples(20), _fixed_metric(1.0), budget=budget
    )

    assert report.truncated is True
    assert report.n < 20
    # cost_used must stay within the cap + at most one in-flight cost per slot
    assert budget.cost_used < budget.max_cost_usd + concurrency * cost_per_example


# ---------------------------------------------------------------------------
# Finding 2: crashed example isolation (TDD)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crashed_example_does_not_kill_eval() -> None:
    """An LLMError from program.run on one example is isolated; others still score."""
    # With concurrency=1 (sequential) the 5th LLM call (example index 4) raises.
    client = _FailOnNthCallClient(fail_on_call=5)
    harness = EvalHarness(client, _cfg(), concurrency=1)
    report = await harness.evaluate(_program(), _candidate(), _examples(10), _fixed_metric(1.0))

    assert report.n == 10
    failed = [r for r in report.per_example if r.failed]
    assert len(failed) == 1
    assert "error:" in failed[0].feedback
    # All other results should be scored normally.
    assert sum(1 for r in report.per_example if not r.failed) == 9


# ---------------------------------------------------------------------------
# Metric exception isolation (TDD — bug 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metric_exception_does_not_kill_eval() -> None:
    """A metric that raises on one example is isolated; other examples still score."""
    fake = FakeLLMClient(script=["[[answer]]: ok" for _ in range(4)])

    def metric(example: Example, prediction) -> MetricResult:
        if example.inputs["question"] == "Q2":
            raise RuntimeError("judge network error")
        return MetricResult(score=1.0)

    harness = EvalHarness(fake, _cfg(), concurrency=1)
    report = await harness.evaluate(_program(), _candidate(), _examples(4), metric)

    assert report.n == 4
    failed = [r for r in report.per_example if r.failed]
    assert len(failed) == 1
    assert failed[0].example_idx == 2
    assert failed[0].score == 0.0
    assert "metric error:" in failed[0].feedback
    # The other three examples scored normally.
    assert sum(1 for r in report.per_example if not r.failed) == 3
    assert all(r.score == 1.0 for r in report.per_example if not r.failed)


# ---------------------------------------------------------------------------
# Finding 4 (minor): concurrent rollout wall
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_rollout_wall() -> None:
    """concurrency=4, max_rollouts=5, 20 examples → exactly 5 results."""
    fake = FakeLLMClient(script=["[[answer]]: ok"] * 20)
    budget = Budget(max_rollouts=5)
    harness = EvalHarness(fake, _cfg(), concurrency=4)
    report = await harness.evaluate(
        _program(), _candidate(), _examples(20), _fixed_metric(1.0), budget=budget
    )
    assert report.n == 5
    assert report.truncated is True


# ---------------------------------------------------------------------------
# Finding 3: Budget.try_reserve / Budget.add_cost API (TDD)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_try_reserve_succeeds_when_not_exhausted() -> None:
    b = Budget(max_rollouts=5)
    result = await b.try_reserve(rollouts=1)
    assert result is True
    assert b.rollouts_used == 1


@pytest.mark.asyncio
async def test_budget_try_reserve_fails_when_exhausted() -> None:
    b = Budget(max_rollouts=2)
    await b.try_reserve(rollouts=1)
    await b.try_reserve(rollouts=1)
    # now exhausted
    result = await b.try_reserve(rollouts=1)
    assert result is False
    assert b.rollouts_used == 2  # not incremented


def test_budget_add_cost() -> None:
    b = Budget(max_cost_usd=5.0)
    b.add_cost(2.5)
    assert b.cost_used == pytest.approx(2.5)
    assert not b.exhausted
    b.add_cost(2.5)
    assert b.exhausted


def test_budget_exhausted_exported() -> None:
    """BudgetExhausted must be importable from promptline.eval.harness."""
    exc = BudgetExhausted("over budget")
    assert isinstance(exc, RuntimeError)
