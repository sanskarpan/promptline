"""Tests for the MIPRO-like Bayesian optimizer."""

from __future__ import annotations

import pytest

from promptline.core.llm import FakeLLMClient, LLMCall
from promptline.core.program import ModelConfig, PromptProgram
from promptline.core.types import Candidate, Example, ModuleState
from promptline.eval.harness import Budget, EvalHarness, MetricResult
from promptline.optimizers.base import RunEvent
from promptline.optimizers.mipro import MIPRO, TIPS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MARKER = "ULTRA_PRECISE_MODE"
SUMMARY_TEXT = "- Questions ask for capital cities.\n- Answers are single words."


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


def _marker_client(marker_proposal: int = 2) -> FakeLLMClient:
    """Scripted client for the marker scenario.

    - Dataset-summary calls return SUMMARY_TEXT.
    - Instruction-proposal call number *marker_proposal* returns an instruction
      containing MARKER; all others return generic instructions.
    - Task calls emit a good answer iff the system prompt contains MARKER.
    """
    counter = {"proposals": 0}

    def _respond(call: LLMCall) -> str:
        last = call.messages[-1].content
        if "Summarize the patterns" in last:
            return SUMMARY_TEXT
        if "Write an improved instruction" in last:
            counter["proposals"] += 1
            if counter["proposals"] == marker_proposal:
                return f"```\nAlways answer in {MARKER} style.\n```"
            return f"```\nGeneric instruction {counter['proposals']}.\n```"
        # Task call: quality depends on the system prompt.
        if MARKER in call.messages[0].content:
            return "[[answer]]: GOOD"
        return "[[answer]]: BAD"

    return FakeLLMClient(script=_respond)


def _good_metric(example: Example, prediction) -> MetricResult:  # type: ignore[type-arg]
    got = prediction.outputs.get("answer", "")
    return MetricResult(score=1.0 if got.strip() == "GOOD" else 0.0)


def _always_pass_metric(example: Example, prediction) -> MetricResult:  # type: ignore[type-arg]
    return MetricResult(score=1.0)


def _pass_client() -> FakeLLMClient:
    """Client where task calls always parse; proposals return fenced text."""
    counter = {"proposals": 0}

    def _respond(call: LLMCall) -> str:
        last = call.messages[-1].content
        if "Summarize the patterns" in last:
            return SUMMARY_TEXT
        if "Write an improved instruction" in last:
            counter["proposals"] += 1
            return f"```\nInstruction variant {counter['proposals']}.\n```"
        return "[[answer]]: Paris"

    return FakeLLMClient(script=_respond)


def _examples(n: int) -> list[Example]:
    return [
        Example(inputs={"question": f"unique-question-{i}"}, labels={"answer": "GOOD"})
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Marker scenario: TPE must find the winning instruction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mipro_finds_marker_instruction() -> None:
    """Instruction candidate #2 (contains MARKER) is strictly best; TPE finds it."""
    program = _program()
    seed = _seed(program)
    client = _marker_client(marker_proposal=2)
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=500)

    opt = MIPRO(
        n_instruction_candidates=4,
        n_demo_sets=2,
        demos_per_set=2,
        n_trials=30,
        minibatch_size=2,
        rng_seed=0,
    )
    result = await opt.optimize(program, seed, _examples(6), _good_metric, budget, harness)

    best_instruction = result.best.modules["main"].instruction
    assert MARKER in best_instruction, (
        f"Expected MARKER in best instruction, got: {best_instruction!r}"
    )
    # Full-eval score of the marker config is 1.0 (all examples GOOD).
    assert result.scores[result.best.id] == pytest.approx(1.0)
    # Lineage: child of seed, optimizer tag, config meta recorded.
    assert result.best.parent_ids == [seed.id]
    assert result.best.optimizer == "mipro"
    assert result.best.meta["inst"]["main"] == 2
    assert "demo" in result.best.meta


# ---------------------------------------------------------------------------
# Grounded proposal prompts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mipro_summary_and_proposal_prompts() -> None:
    """Summary prompt shows sampled examples; proposal prompts show summary + tip."""
    program = _program()
    seed = _seed(program)
    client = _marker_client()
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=500)

    opt = MIPRO(
        n_instruction_candidates=4,
        n_demo_sets=2,
        n_trials=5,
        minibatch_size=2,
        rng_seed=0,
    )
    await opt.optimize(program, seed, _examples(5), _good_metric, budget, harness)

    summary_calls = [c for c in client.calls if "Summarize the patterns" in c.messages[-1].content]
    assert len(summary_calls) == 1
    summary_prompt = summary_calls[0].messages[-1].content
    # All 5 examples fit in the 10-example sample, so each must appear.
    for i in range(5):
        assert f"unique-question-{i}" in summary_prompt
    assert "GOOD" in summary_prompt  # labels rendered too

    proposal_calls = [
        c for c in client.calls if "Write an improved instruction" in c.messages[-1].content
    ]
    assert len(proposal_calls) == 3  # n_instruction_candidates - 1
    for call in proposal_calls:
        content = call.messages[-1].content
        assert SUMMARY_TEXT in content, "proposal prompt must include dataset summary"
        assert any(tip in content for tip in TIPS), "proposal prompt must include a tip"
        assert call.temperature == 1.0
    # Distinct seeds on proposer calls.
    seeds = [c.seed for c in proposal_calls]
    assert len(set(seeds)) == len(seeds)


# ---------------------------------------------------------------------------
# Demo sets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mipro_demo_sets_zero_shot_and_capped() -> None:
    """Demo set 0 is empty (zero-shot); other sets have at most demos_per_set demos."""
    program = _program()
    seed = _seed(program)
    client = _pass_client()
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=500)

    demos_per_set = 2
    opt = MIPRO(
        n_instruction_candidates=2,
        n_demo_sets=3,
        demos_per_set=demos_per_set,
        n_trials=20,
        minibatch_size=2,
        rng_seed=0,
    )
    result = await opt.optimize(program, seed, _examples(8), _always_pass_metric, budget, harness)

    trial_candidates = [c for c in result.candidates if c.id != seed.id]
    assert trial_candidates, "expected at least one trial candidate"
    saw_zero_shot = False
    for cand in trial_candidates:
        demo_idx = cand.meta["demo"]["main"]
        demos = cand.modules["main"].demos
        assert len(demos) <= demos_per_set
        if demo_idx == 0:
            saw_zero_shot = True
            assert demos == []
    assert saw_zero_shot, "TPE never sampled the zero-shot demo set in 20 trials"


# ---------------------------------------------------------------------------
# Config dedupe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mipro_config_dedupe_no_reeval() -> None:
    """The same config sampled twice must not be re-evaluated."""
    program = _program()
    seed = _seed(program)
    client = _pass_client()
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=500)

    # Exactly one possible config (original instruction, zero-shot set only).
    opt = MIPRO(
        n_instruction_candidates=1,
        n_demo_sets=1,
        n_trials=5,
        minibatch_size=4,
        full_eval_steps=100,  # no mid-run full evals
        rng_seed=0,
    )
    events: list[RunEvent] = []
    trainset = _examples(4)
    result = await opt.optimize(
        program,
        seed,
        trainset,
        _always_pass_metric,
        budget,
        harness,
        emit=events.append,
    )

    # No pool collection (single, empty demo set), no proposals.  Rollouts:
    # one minibatch eval (4) + one final full eval (4) = 8 despite 5 trials.
    assert budget.rollouts_used == 8

    scored = [e for e in events if e.type == "minibatch_scored"]
    assert len(scored) == 5
    assert sum(1 for e in scored if not e.payload["cached"]) == 1
    assert sum(1 for e in scored if e.payload["cached"]) == 4
    assert result.best is not None


# ---------------------------------------------------------------------------
# Budget discipline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mipro_budget_early_stop() -> None:
    """A tiny rollout budget stops the run early without crashing."""
    program = _program()
    seed = _seed(program)
    client = _marker_client()
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=3)

    opt = MIPRO(n_trials=30, minibatch_size=4, rng_seed=0)
    result = await opt.optimize(program, seed, _examples(10), _good_metric, budget, harness)

    assert result.best is not None
    assert result.best.id in result.scores
    assert budget.rollouts_used <= 3


# ---------------------------------------------------------------------------
# Events and determinism
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mipro_truncated_full_eval_not_recorded() -> None:
    """A budget-truncated full eval must not be stored or emitted."""
    program = _program()
    seed = _seed(program)
    client = _pass_client()
    harness = EvalHarness(client=client, cfg=_model_cfg())
    # Minibatch (2 rollouts) fits; the full eval over 6 examples truncates at 5.
    budget = Budget(max_rollouts=5)

    opt = MIPRO(
        n_instruction_candidates=1,
        n_demo_sets=1,
        n_trials=5,
        minibatch_size=2,
        full_eval_steps=1,
        rng_seed=0,
    )
    events: list[RunEvent] = []
    result = await opt.optimize(
        program,
        seed,
        _examples(6),
        _always_pass_metric,
        budget,
        harness,
        emit=events.append,
    )

    assert all(e.type != "full_eval" for e in events), (
        "truncated full eval must not emit a full_eval event"
    )
    # Best falls back to the minibatch score, untainted by the truncated mean.
    assert result.scores[result.best.id] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_mipro_truncated_minibatch_pruned_not_scored() -> None:
    """A budget-truncated minibatch prunes the trial without recording a score."""
    program = _program()
    seed = _seed(program)
    client = _pass_client()
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=1)  # minibatch of 4 truncates immediately

    opt = MIPRO(
        n_instruction_candidates=1,
        n_demo_sets=1,
        n_trials=5,
        minibatch_size=4,
        full_eval_steps=100,
        rng_seed=0,
    )
    events: list[RunEvent] = []
    result = await opt.optimize(
        program,
        seed,
        _examples(6),
        _always_pass_metric,
        budget,
        harness,
        emit=events.append,
    )

    assert all(e.type != "minibatch_scored" for e in events), (
        "truncated minibatch must not emit a minibatch_scored event"
    )
    # No config survived: the seed is returned as best.
    assert result.best.id == seed.id


@pytest.mark.asyncio
async def test_mipro_events_all_types() -> None:
    """A full run emits all six event types in a sane order."""
    program = _program()
    seed = _seed(program)
    client = _marker_client()
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=500)

    events: list[RunEvent] = []
    opt = MIPRO(
        n_instruction_candidates=4,
        n_demo_sets=2,
        n_trials=12,
        minibatch_size=2,
        full_eval_steps=5,
        rng_seed=0,
    )
    await opt.optimize(
        program,
        seed,
        _examples(6),
        _good_metric,
        budget,
        harness,
        emit=events.append,
    )

    types = [e.type for e in events]
    assert types[0] == "run_started"
    assert types[-1] == "run_finished"
    for expected in (
        "candidate_proposed",
        "minibatch_scored",
        "full_eval",
        "budget_tick",
    ):
        assert expected in types, f"missing event type {expected}"

    proposed = [e for e in events if e.type == "candidate_proposed"]
    assert all("module" in e.payload and "tip" in e.payload for e in proposed)
    scored = [e for e in events if e.type == "minibatch_scored"]
    assert all(
        "trial" in e.payload and "config" in e.payload and "score" in e.payload for e in scored
    )


@pytest.mark.asyncio
async def test_mipro_deterministic_same_seed() -> None:
    """Two runs with the same rng_seed pick the same best instruction."""

    async def _run() -> str:
        program = _program()
        seed = _seed(program)
        client = _marker_client()
        harness = EvalHarness(client=client, cfg=_model_cfg())
        budget = Budget(max_rollouts=500)
        opt = MIPRO(
            n_instruction_candidates=4,
            n_demo_sets=2,
            n_trials=15,
            minibatch_size=2,
            rng_seed=7,
        )
        result = await opt.optimize(program, seed, _examples(6), _good_metric, budget, harness)
        return result.best.modules["main"].instruction

    first = await _run()
    second = await _run()
    assert first == second
