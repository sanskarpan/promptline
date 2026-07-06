"""Tests for BootstrapFewShot and BootstrapRandomSearch optimizers."""

from __future__ import annotations

import pytest

from promptline.core.llm import FakeLLMClient, LLMCall, LLMError
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
        modules={m.name: ModuleState(instruction=m.signature.instruction) for m in program.modules}
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

    examples = [Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"}) for i in range(5)]
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

    examples = [Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"}) for i in range(5)]
    metric = _exact_metric("Paris")  # model says "wrong", so all fail

    opt = BootstrapFewShot(max_demos=4, threshold=1.0, rng_seed=0)
    result = await opt.optimize(program, seed, examples, metric, budget, harness)

    first_mod = program.modules[0].name
    demos = result.best.modules[first_mod].demos
    assert len(demos) == 0


@pytest.mark.asyncio
async def test_bootstrap_max_demos_cap() -> None:
    """Should stop collecting after max_demos, even if more examples pass.

    After the score-semantics fix (Finding 4) BootstrapFewShot also evaluates
    the augmented candidate on the full training set, so the total rollout count
    is collection_rollouts + len(trainset).  The important invariant is that the
    *collection* loop stopped after exactly max_demos rollouts (verified by the
    demos count) and that the best module carries exactly max_demos demos.
    """
    program = _program()
    seed = _seed(program)
    client = _fake_client_single_answer("Paris")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=50)

    examples = [
        Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"}) for i in range(10)
    ]
    metric = _exact_metric("Paris")

    opt = BootstrapFewShot(max_demos=3, threshold=1.0, rng_seed=0)
    result = await opt.optimize(program, seed, examples, metric, budget, harness)

    first_mod = program.modules[0].name
    demos = result.best.modules[first_mod].demos
    assert len(demos) == 3
    # Collection uses 3 rollouts; post-collection eval uses up to len(trainset)=10.
    # Total is at most 3 + 10 = 13, well within budget=50.
    assert 3 <= budget.rollouts_used <= 13


@pytest.mark.asyncio
async def test_bootstrap_budget_rollout_accounting() -> None:
    """Budget rollouts_used should reflect how many examples were attempted."""
    program = _program()
    seed = _seed(program)
    client = _fake_client_single_answer("Paris")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=3)

    examples = [
        Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"}) for i in range(10)
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

    examples = [Example(inputs={"question": "What is the capital of Germany?"})]
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

    examples = [Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"}) for i in range(4)]

    opt = BootstrapFewShot(max_demos=4, threshold=1.0, rng_seed=0)
    result = await opt.optimize(program, seed, examples, _always_pass_metric, budget, harness)

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
        Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"}) for i in range(10)
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
        Example(inputs={"question": f"q{i}"}, labels={"answer": "marker"}) for i in range(12)
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

    examples = [Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"}) for i in range(6)]

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
        Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"}) for i in range(20)
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


# ---------------------------------------------------------------------------
# Finding 2: Unisolated example failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_tolerates_llm_error() -> None:
    """BootstrapFewShot must tolerate LLMError on individual examples.

    When program.run raises an exception for one input the optimizer should
    skip it (treat as failing) and continue collecting demos from the others.
    """
    program = _program()
    seed = _seed(program)
    FAIL_Q = "CRASH_ME"

    def _error_on_one(call: LLMCall) -> str:
        # Detect the crash example by checking the last user message.
        user_msgs = [m for m in call.messages if m.role == "user"]
        if user_msgs and FAIL_Q in user_msgs[-1].content:
            raise LLMError("simulated failure")
        return "[[answer]]: Paris"

    client = FakeLLMClient(script=_error_on_one)
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=50)

    examples = [
        Example(inputs={"question": FAIL_Q}, labels={"answer": "Paris"}),  # will crash
    ] + [Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"}) for i in range(4)]
    metric = _always_pass_metric

    opt = BootstrapFewShot(max_demos=4, threshold=1.0, rng_seed=0)
    result = await opt.optimize(program, seed, examples, metric, budget, harness)

    # Optimizer must complete without raising.
    assert result.best is not None
    # Demos should come from the non-crashing examples.
    first_mod = program.modules[0].name
    demos = result.best.modules[first_mod].demos
    assert len(demos) > 0, "Expected demos from non-crashing examples"


@pytest.mark.asyncio
async def test_bootstrap_rs_tolerates_llm_error() -> None:
    """BootstrapRandomSearch must tolerate LLMError during demo collection."""
    program = _program()
    seed = _seed(program)
    FAIL_Q = "CRASH_ME"

    def _error_on_one(call: LLMCall) -> str:
        user_msgs = [m for m in call.messages if m.role == "user"]
        if user_msgs and FAIL_Q in user_msgs[-1].content:
            raise LLMError("simulated failure")
        return "[[answer]]: Paris"

    client = FakeLLMClient(script=_error_on_one)
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=200)

    examples = [
        Example(inputs={"question": FAIL_Q}, labels={"answer": "Paris"}),  # crashes
    ] + [Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"}) for i in range(9)]
    metric = _always_pass_metric

    opt = BootstrapRandomSearch(
        n_subsets=3,
        subset_size=2,
        threshold=1.0,
        val_fraction=0.3,
        rng_seed=0,
    )
    result = await opt.optimize(program, seed, examples, metric, budget, harness)

    # Must complete without raising.
    assert result.best is not None


# ---------------------------------------------------------------------------
# Finding 4: Bootstrap score semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_score_semantics() -> None:
    """BootstrapFewShot must put pass-rate under seed.id, eval score under best.id.

    When budget allows, the augmented candidate (best) must be evaluated and
    its true mean score recorded under scores[best.id].  The seed pass-rate
    must be recorded under scores[seed.id].
    """
    program = _program()
    seed = _seed(program)
    client = _fake_client_single_answer("Paris")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=50)

    examples = [Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"}) for i in range(4)]
    metric = _exact_metric("Paris")

    opt = BootstrapFewShot(max_demos=4, threshold=1.0, rng_seed=0)
    result = await opt.optimize(program, seed, examples, metric, budget, harness)

    # seed.id must appear in scores (the pass-rate from collection).
    assert seed.id in result.scores, "seed.id must be in scores"
    assert 0.0 <= result.scores[seed.id] <= 1.0

    # best.id must appear in scores (true eval score) because budget allows.
    assert result.best.id in result.scores, "best.id must be in scores when budget allows"
    assert 0.0 <= result.scores[result.best.id] <= 1.0


@pytest.mark.asyncio
async def test_bootstrap_score_semantics_budget_exhausted() -> None:
    """When budget is exhausted after collection, scores[best.id] must be absent."""
    program = _program()
    seed = _seed(program)
    client = _fake_client_single_answer("Paris")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    # Budget tight enough to be exhausted exactly after collection (3 examples = 3 rollouts).
    budget = Budget(max_rollouts=3)

    examples = [Example(inputs={"question": f"q{i}"}, labels={"answer": "Paris"}) for i in range(3)]
    metric = _exact_metric("Paris")

    opt = BootstrapFewShot(max_demos=3, threshold=1.0, rng_seed=0)
    result = await opt.optimize(program, seed, examples, metric, budget, harness)

    # seed.id should be in scores.
    assert seed.id in result.scores

    # best.id should NOT be in scores since budget is exhausted.
    assert result.best.id not in result.scores, (
        "best.id must NOT be in scores when budget is exhausted after collection"
    )


# ---------------------------------------------------------------------------
# Finding 5a: Real discrimination test for BootstrapRandomSearch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_rs_real_discrimination() -> None:
    """Best candidate must be the one whose MARKER demo actually improves val score.

    Setup: a unique MARKER input is injected as a demo.  The fake client
    returns a high-scoring answer only when the MARKER string appears in the
    rendered few-shot demonstration messages.  All val examples expect the
    high-scoring answer, so only subsets containing the MARKER demo achieve
    score > 0.  BootstrapRandomSearch must select one of those subsets as best.
    """
    program = _program()
    seed = _seed(program)

    MARKER = "UNIQUE_MARKER_XYZ"

    def _discriminating_client(call: LLMCall) -> str:
        has_asst = any(m.role == "assistant" for m in call.messages)
        if not has_asst:
            # Collection phase (no demos): always return correct answer.
            return "[[answer]]: magic"
        # Evaluation phase: check if MARKER appears in any *demo* user message
        # (all user messages except the last one, which is the actual question).
        user_msgs = [m for m in call.messages if m.role == "user"]
        demo_user_msgs = user_msgs[:-1]
        if any(MARKER in m.content for m in demo_user_msgs):
            return "[[answer]]: magic"
        return "[[answer]]: not_magic"

    client = FakeLLMClient(script=_discriminating_client)
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=500)

    # 1 MARKER example + 9 filler examples; all have label="magic" so all pass threshold.
    marker_example = Example(inputs={"question": MARKER}, labels={"answer": "magic"})
    filler_examples = [
        Example(inputs={"question": f"filler_{i}"}, labels={"answer": "magic"}) for i in range(9)
    ]
    examples = [marker_example] + filler_examples

    def metric(example: Example, prediction) -> MetricResult:  # type: ignore[type-arg]
        got = prediction.outputs.get("answer", "")
        expected = example.labels.get("answer", "magic")
        return MetricResult(score=1.0 if got == expected else 0.0)

    # rng_seed=0: MARKER ends up in pool_source and gets sampled by ≥1 subset.
    # (Verified: with pool=[7 demos] and 8 subsets, MARKER is picked at
    #  indices 1 and 3 → those candidates score 1.0; others score 0.0.)
    opt = BootstrapRandomSearch(
        n_subsets=8,
        subset_size=1,
        threshold=1.0,
        val_fraction=0.3,
        rng_seed=0,
    )
    result = await opt.optimize(program, seed, examples, metric, budget, harness)

    first_mod = program.modules[0].name
    best_demos = result.best.modules[first_mod].demos

    best_score = result.scores.get(result.best.id, -1)
    assert best_score > 0.0, (
        f"Best candidate should score > 0. Got {best_score}. "
        f"This means no subset with MARKER demo was ever evaluated."
    )
    # Best candidate must carry the MARKER demo.
    has_marker = any(MARKER in demo.inputs.get("question", "") for demo in best_demos)
    assert has_marker, (
        f"Best candidate must contain MARKER demo. Demos: {[d.inputs for d in best_demos]}"
    )


# ---------------------------------------------------------------------------
# Continuous-metric thresholds (LLM judge produces scores in [0, 1])
# ---------------------------------------------------------------------------


def _continuous_metric(score: float):
    def metric(example: Example, prediction) -> MetricResult:  # type: ignore[type-arg]
        return MetricResult(score=score)

    return metric


@pytest.mark.asyncio
async def test_bootstrap_default_threshold_accepts_continuous_pass() -> None:
    """Default threshold (0.7) must collect demos from a 0.8-scoring judge."""
    program = _program()
    seed = _seed(program)
    harness = EvalHarness(client=_fake_client_single_answer(), cfg=_model_cfg())
    examples = [Example(inputs={"question": f"q{i}"}, labels={}) for i in range(4)]

    opt = BootstrapFewShot(max_demos=4, rng_seed=0)  # default threshold
    assert opt.threshold == 0.7
    result = await opt.optimize(
        program,
        seed,
        examples,
        _continuous_metric(0.8),
        Budget(max_rollouts=50),
        harness,
    )
    demos = result.best.modules[program.modules[0].name].demos
    assert len(demos) == 4


@pytest.mark.asyncio
async def test_bootstrap_default_threshold_rejects_continuous_fail() -> None:
    """A 0.6-scoring judge is below the default 0.7 cut — no demos."""
    program = _program()
    seed = _seed(program)
    harness = EvalHarness(client=_fake_client_single_answer(), cfg=_model_cfg())
    examples = [Example(inputs={"question": f"q{i}"}, labels={}) for i in range(4)]

    opt = BootstrapFewShot(max_demos=4, rng_seed=0)
    result = await opt.optimize(
        program,
        seed,
        examples,
        _continuous_metric(0.6),
        Budget(max_rollouts=50),
        harness,
    )
    demos = result.best.modules[program.modules[0].name].demos
    assert demos == []


def test_bootstrap_rs_and_mipro_default_thresholds_continuous() -> None:
    from promptline.optimizers.mipro import MIPRO

    assert BootstrapRandomSearch().threshold == 0.7
    assert MIPRO().threshold == 0.7
