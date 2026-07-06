"""Cross-optimizer event payload consistency.

All producers of ``candidate_proposed`` emit ``parents: [ids...]``; every
optimizer's ``run_finished`` carries ``best_score`` when the best is known;
``budget_tick`` events include the budget ceilings.
"""

from __future__ import annotations

import pytest

from promptline.core.llm import FakeLLMClient, LLMCall
from promptline.core.program import ModelConfig, PromptProgram
from promptline.core.types import Candidate, Example, ModuleState
from promptline.eval.harness import Budget, EvalHarness, MetricResult
from promptline.optimizers.base import RunEvent

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


def _examples(n: int = 4) -> list[Example]:
    return [Example(inputs={"question": f"q{i}"}, labels={}) for i in range(n)]


def _always_pass_metric(example: Example, prediction) -> MetricResult:  # type: ignore[type-arg]
    return MetricResult(score=1.0)


def _multi_client() -> FakeLLMClient:
    """Answers task calls; proposer/reflection calls get plausible payloads."""

    def _respond(call: LLMCall) -> str:
        content = call.messages[-1].content if call.messages else ""
        if "Write a new instruction" in content:
            return "<INS>New improved instruction.</INS>"
        if "fenced code block" in content or "Paraphrase" in content:
            return "```\nBetter instruction.\n```"
        if "diagnose" in content:
            return "The instruction is too vague."
        if "Summarize the patterns" in content:
            return "- pattern"
        return "[[answer]]: fine"

    return FakeLLMClient(script=_respond)


def _collect(events: list[RunEvent], type_: str) -> list[RunEvent]:
    return [e for e in events if e.type == type_]


async def _run(opt, metric=_always_pass_metric, budget: Budget | None = None):
    program = _program()
    seed = _seed(program)
    harness = EvalHarness(client=_multi_client(), cfg=ModelConfig(task_model="fake"))
    events: list[RunEvent] = []
    budget = budget or Budget(max_rollouts=60)
    result = await opt.optimize(
        program, seed, _examples(), metric, budget, harness, emit=events.append
    )
    return result, events, seed


def _assert_common(result, events, seed, expect_parents: bool = True) -> None:
    finished = _collect(events, "run_finished")
    assert finished, "no run_finished emitted"
    payload = finished[-1].payload
    assert payload["best_id"] == result.best.id
    assert payload.get("best_score") == result.scores.get(result.best.id)

    ticks = _collect(events, "budget_tick")
    assert ticks, "no budget_tick emitted"
    for tick in ticks:
        assert "max_rollouts" in tick.payload
        assert "max_cost_usd" in tick.payload
        assert "rollouts_used" in tick.payload

    if expect_parents:
        proposed = _collect(events, "candidate_proposed")
        assert proposed, "no candidate_proposed emitted"
        for event in proposed:
            parents = event.payload.get("parents")
            assert isinstance(parents, list) and parents, (
                f"candidate_proposed without parents: {event.payload}"
            )


# ---------------------------------------------------------------------------
# Per-optimizer checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_event_payloads() -> None:
    from promptline.optimizers.bootstrap import BootstrapFewShot

    result, events, seed = await _run(BootstrapFewShot(rng_seed=0))
    _assert_common(result, events, seed)
    proposed = _collect(events, "candidate_proposed")
    assert proposed[0].payload["parents"] == [seed.id]


@pytest.mark.asyncio
async def test_bootstrap_rs_event_payloads() -> None:
    from promptline.optimizers.bootstrap import BootstrapRandomSearch

    result, events, seed = await _run(BootstrapRandomSearch(n_subsets=2, subset_size=2, rng_seed=0))
    _assert_common(result, events, seed)


@pytest.mark.asyncio
async def test_opro_event_payloads() -> None:
    from promptline.optimizers.opro import OPRO

    result, events, seed = await _run(OPRO(n_steps=2, candidates_per_step=1, minibatch_size=2))
    _assert_common(result, events, seed)
    proposed = _collect(events, "candidate_proposed")
    assert all(e.payload["parents"] == [seed.id] for e in proposed)


@pytest.mark.asyncio
async def test_mipro_event_payloads() -> None:
    from promptline.optimizers.mipro import MIPRO

    def _mixed_metric(example: Example, prediction) -> MetricResult:  # type: ignore[type-arg]
        return MetricResult(score=0.9)

    result, events, seed = await _run(
        MIPRO(
            n_instruction_candidates=2,
            n_demo_sets=2,
            demos_per_set=1,
            n_trials=3,
            minibatch_size=2,
            rng_seed=0,
        ),
        metric=_mixed_metric,
    )
    _assert_common(result, events, seed)
    # minibatch_scored carries both `score` and the `mean_score` alias + trial.
    scored = _collect(events, "minibatch_scored")
    assert scored
    for event in scored:
        assert event.payload["mean_score"] == event.payload["score"]
        assert "trial" in event.payload


@pytest.mark.asyncio
async def test_protegi_event_payloads() -> None:
    from promptline.optimizers.protegi import ProTeGi

    def _failing_metric(example: Example, prediction) -> MetricResult:  # type: ignore[type-arg]
        # Below the failure threshold → gradients are generated.
        return MetricResult(score=0.5, feedback="too vague")

    result, events, _ = await _run(
        ProTeGi(
            beam_width=2,
            n_gradients=1,
            n_paraphrases=1,
            n_rounds=1,
            minibatch_size=2,
            racing_rounds=1,
            racing_batch=2,
        ),
        metric=_failing_metric,
    )
    _assert_common(result, events, None)
    # Legacy parent_id alias is preserved alongside parents.
    proposed = _collect(events, "candidate_proposed")
    for event in proposed:
        assert event.payload["parents"] == [event.payload["parent_id"]]


@pytest.mark.asyncio
async def test_gepa_run_finished_has_best_score() -> None:
    from promptline.optimizers.gepa import GEPA

    result, events, _ = await _run(GEPA(max_iterations=2, minibatch_size=2, n_pareto=2))
    finished = _collect(events, "run_finished")
    assert finished
    assert finished[-1].payload["best_score"] == result.scores[result.best.id]
    # GEPA mutation proposals emit parents alongside the legacy parent_id.
    for event in _collect(events, "candidate_proposed"):
        if "parent_id" in event.payload:
            assert event.payload["parents"] == [event.payload["parent_id"]]
        else:
            assert event.payload.get("parents"), event.payload
