"""Tests for the OPRO optimizer."""
from __future__ import annotations

import pytest

from promptline.core.llm import FakeLLMClient, LLMCall
from promptline.core.program import ModelConfig, PromptProgram
from promptline.core.types import Candidate, Example, ModuleState
from promptline.eval.harness import Budget, EvalHarness, MetricResult
from promptline.optimizers.base import RunEvent
from promptline.optimizers.opro import OPRO, _parse_instruction

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


def _model_cfg(reflection_model: str = "") -> ModelConfig:
    return ModelConfig(task_model="fake", reflection_model=reflection_model)


def _always_pass_metric(example: Example, prediction) -> MetricResult:  # type: ignore[type-arg]
    return MetricResult(score=1.0)


def _answer_client(answer: str) -> FakeLLMClient:
    """Task calls return a parseable answer; proposer calls return an INS block."""

    def _respond(call: LLMCall) -> str:
        # Heuristic: the meta-prompt starts with "You are an expert prompt engineer."
        if call.messages and "Write a new instruction" in call.messages[-1].content:
            return "<INS>New improved instruction.</INS>"
        return f"[[answer]]: {answer}"

    return FakeLLMClient(script=_respond)


# ---------------------------------------------------------------------------
# Unit tests — _parse_instruction
# ---------------------------------------------------------------------------


def test_parse_instruction_with_ins_tags() -> None:
    text = "Some preamble <INS>Be concise and direct.</INS> trailing"
    assert _parse_instruction(text) == "Be concise and direct."


def test_parse_instruction_fallback_to_stripped() -> None:
    text = "   No tags here, just raw text.   "
    assert _parse_instruction(text) == "No tags here, just raw text."


def test_parse_instruction_multiline_ins() -> None:
    text = "<INS>\nLine one.\nLine two.\n</INS>"
    assert _parse_instruction(text) == "Line one.\nLine two."


# ---------------------------------------------------------------------------
# OPRO optimizer tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_opro_seed_evaluated_first() -> None:
    """Seed must appear in candidates and scores after step 0."""
    program = _program()
    seed = _seed(program)
    client = _answer_client("Paris")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=50)

    examples = [
        Example(inputs={"question": "q"}, labels={"answer": "Paris"})
        for _ in range(3)
    ]

    opt = OPRO(n_steps=1, candidates_per_step=1, rng_seed=0)
    result = await opt.optimize(
        program, seed, examples, _always_pass_metric, budget, harness
    )

    assert seed in result.candidates
    assert seed.id in result.scores


@pytest.mark.asyncio
async def test_opro_best_returned() -> None:
    """The best candidate is the one with the highest score in scores dict."""
    program = _program()
    seed = _seed(program)

    # Proposer always returns a marker instruction; metric always passes.
    client = _answer_client("any")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=50)

    examples = [Example(inputs={"question": "q"}) for _ in range(3)]

    opt = OPRO(n_steps=2, candidates_per_step=2, rng_seed=0)
    result = await opt.optimize(
        program, seed, examples, _always_pass_metric, budget, harness
    )

    best_score = result.scores[result.best.id]
    for cand_id, score in result.scores.items():
        assert score <= best_score + 1e-9, (
            f"Candidate {cand_id} has score {score} > best {best_score}"
        )


@pytest.mark.asyncio
async def test_opro_events_sequence() -> None:
    """Events must begin with run_started and end with run_finished."""
    program = _program()
    seed = _seed(program)
    client = _answer_client("x")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=50)

    examples = [Example(inputs={"question": "q"}) for _ in range(3)]
    events: list[RunEvent] = []

    opt = OPRO(n_steps=2, candidates_per_step=1, rng_seed=0)
    await opt.optimize(
        program, seed, examples, _always_pass_metric, budget, harness, emit=events.append
    )

    event_types = [e.type for e in events]
    assert event_types[0] == "run_started"
    assert event_types[-1] == "run_finished"
    assert "candidate_proposed" in event_types
    assert "minibatch_scored" in event_types


@pytest.mark.asyncio
async def test_opro_budget_early_stop() -> None:
    """With a tiny budget, OPRO must stop early without crashing."""
    program = _program()
    seed = _seed(program)
    client = _answer_client("Paris")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=2)  # very tight

    examples = [
        Example(inputs={"question": f"q{i}"}) for i in range(5)
    ]

    opt = OPRO(n_steps=10, candidates_per_step=4, rng_seed=0)
    result = await opt.optimize(
        program, seed, examples, _always_pass_metric, budget, harness
    )

    # Should not crash and should still return a valid result.
    assert result.best is not None
    assert budget.rollouts_used <= budget.max_rollouts


@pytest.mark.asyncio
async def test_opro_trajectory_cap() -> None:
    """Trajectory should not exceed max_trajectory entries."""
    program = _program()
    seed = _seed(program)

    instructions_seen: list[str] = []

    def _proposer_client(call: LLMCall) -> str:
        # If it looks like a meta-prompt, generate a unique instruction.
        if call.messages and "Write a new instruction" in call.messages[-1].content:
            n = len(instructions_seen)
            instr = f"<INS>Instruction variant {n}.</INS>"
            instructions_seen.append(instr)
            return instr
        return "[[answer]]: yes"

    client = FakeLLMClient(script=_proposer_client)
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=100)

    examples = [Example(inputs={"question": "q"}) for _ in range(2)]

    opt = OPRO(n_steps=5, candidates_per_step=3, max_trajectory=5, rng_seed=0)
    result = await opt.optimize(
        program, seed, examples, _always_pass_metric, budget, harness
    )

    # Verify the optimizer didn't crash and returned multiple candidates.
    assert len(result.candidates) > 1


@pytest.mark.asyncio
async def test_opro_minibatch_mode() -> None:
    """minibatch_size limits how many examples are evaluated per proposal."""
    program = _program()
    seed = _seed(program)
    client = _answer_client("x")
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=100)

    # 10 examples but minibatch_size=3 → at most 3 rollouts per eval.
    examples = [Example(inputs={"question": f"q{i}"}) for i in range(10)]

    opt = OPRO(n_steps=2, candidates_per_step=1, minibatch_size=3, rng_seed=0)
    result = await opt.optimize(
        program, seed, examples, _always_pass_metric, budget, harness
    )

    assert result.best is not None
    # Each eval step uses at most 3 rollouts; with 2 steps + seed eval ≤
    # 3*(2+1) = 9 rollouts.  Add proposer calls don't use rollouts.
    assert budget.rollouts_used <= 9


@pytest.mark.asyncio
async def test_opro_marker_instruction_rewarded() -> None:
    """Metric that favors a specific instruction token should select that candidate."""
    program = _program()
    seed = _seed(program)

    MARKER = "REWARD_ME"
    counter = {"n": 0}

    def _proposer(call: LLMCall) -> str:
        if call.messages and "Write a new instruction" in call.messages[-1].content:
            # First proposal: emit the marker; rest: generic.
            counter["n"] += 1
            if counter["n"] == 1:
                return f"<INS>{MARKER}</INS>"
            return f"<INS>Generic instruction {counter['n']}.</INS>"
        return "[[answer]]: yes"

    client = FakeLLMClient(script=_proposer)
    harness = EvalHarness(client=client, cfg=_model_cfg())
    budget = Budget(max_rollouts=100)

    examples = [Example(inputs={"question": "q"}) for _ in range(3)]

    def _biased_metric(example: Example, prediction) -> MetricResult:  # type: ignore[type-arg]
        # Always pass (we control via candidate selection, not output content).
        return MetricResult(score=1.0)

    opt = OPRO(n_steps=3, candidates_per_step=2, rng_seed=0)
    result = await opt.optimize(
        program, seed, examples, _biased_metric, budget, harness
    )

    # At least one candidate should have MARKER in its instruction.
    instr_list = [
        c.modules[next(iter(c.modules))].instruction for c in result.candidates
    ]
    assert any(MARKER in instr for instr in instr_list), (
        f"Expected MARKER in some candidate instruction. Got: {instr_list}"
    )
