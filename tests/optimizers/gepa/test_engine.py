"""Integration tests for the GEPA engine."""

from __future__ import annotations

import random
from pathlib import Path

from promptline.core.llm import FakeLLMClient, LLMCall
from promptline.core.program import ModelConfig, Module, Prediction, PromptProgram
from promptline.core.types import Candidate, Example, Field, ModuleState, Signature
from promptline.eval.harness import Budget, EvalHarness, MetricResult
from promptline.optimizers.base import RunEvent
from promptline.optimizers.gepa import GEPA
from promptline.optimizers.gepa.state import GepaState

MARKER = "ALWAYS CITE"

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


def _trainset(n: int = 8) -> list[Example]:
    return [Example(inputs={"question": f"Q{i}?"}) for i in range(n)]


def _marker_metric(example: Example, prediction: Prediction) -> MetricResult:
    """Rewards outputs that contain the citation marker."""
    if MARKER in prediction.outputs.get("answer", ""):
        return MetricResult(score=1.0, feedback="cited sources")
    return MetricResult(score=0.0, feedback="missing citation of sources")


def _client(reflection_instruction: str) -> FakeLLMClient:
    """Task calls echo the marker iff the system prompt contains it.

    Reflection calls (identified by the directive) return
    *reflection_instruction* in a fenced block.
    """

    def _respond(call: LLMCall) -> str:
        joined = "\n".join(m.content for m in call.messages)
        if "Diagnose the failures" in joined:
            return f"The answers never cite.\n```\n{reflection_instruction}\n```"
        system = call.messages[0].content
        if MARKER in system:
            return f"[[answer]]: sources cited. {MARKER}"
        return "[[answer]]: plain answer"

    return FakeLLMClient(script=_respond)


def _harness(client: FakeLLMClient) -> EvalHarness:
    return EvalHarness(client, ModelConfig(task_model="fake"), concurrency=4)


def _collector() -> tuple[list[RunEvent], object]:
    events: list[RunEvent] = []
    return events, events.append


def _always_marker_client() -> FakeLLMClient:
    """Every task call returns the MARKER so _marker_metric scores 1.0 unconditionally."""

    def _respond(call: LLMCall) -> str:
        return f"[[answer]]: always cited. {MARKER}"

    return FakeLLMClient(script=_respond)


# ---------------------------------------------------------------------------
# Improvement loop
# ---------------------------------------------------------------------------


async def test_improvement_loop_accepts_better_child() -> None:
    program = _program()
    seed = _seed(program)
    client = _client(f"Answer the question. {MARKER} sources.")
    result = await GEPA(minibatch_size=2, max_iterations=3, use_merge=False).optimize(
        program,
        seed,
        _trainset(),
        _marker_metric,
        Budget(max_rollouts=100),
        _harness(client),
    )

    assert result.best.id != seed.id
    assert result.scores[result.best.id] > result.scores[seed.id]
    assert result.scores[result.best.id] == 1.0
    assert MARKER in result.best.modules["main"].instruction
    assert result.best.parent_ids  # lineage recorded


# ---------------------------------------------------------------------------
# Strict acceptance
# ---------------------------------------------------------------------------


async def test_strict_acceptance_rejects_non_improving_child() -> None:
    program = _program()
    seed = _seed(program)
    # Reflection proposes an instruction without the marker: child score ==
    # parent score, so strict (>) acceptance must reject it.
    client = _client("Answer the question carefully.")
    result = await GEPA(minibatch_size=2, max_iterations=3, use_merge=False).optimize(
        program,
        seed,
        _trainset(),
        _marker_metric,
        Budget(max_rollouts=100),
        _harness(client),
    )

    assert [c.id for c in result.candidates] == [seed.id]
    assert result.best.id == seed.id


# ---------------------------------------------------------------------------
# Budget wall
# ---------------------------------------------------------------------------


async def test_budget_wall_terminates_within_cap() -> None:
    program = _program()
    seed = _seed(program)
    budget = Budget(max_rollouts=5)
    result = await GEPA(minibatch_size=2, max_iterations=50, use_merge=False).optimize(
        program,
        seed,
        _trainset(),
        _marker_metric,
        budget,
        _harness(_client(f"Improved. {MARKER}.")),
    )

    assert budget.rollouts_used <= 5
    assert result.best is not None


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------


async def test_checkpoint_and_resume(tmp_path: Path) -> None:
    program = _program()
    seed = _seed(program)
    run_dir = tmp_path / "run"
    client = _client(f"Answer the question. {MARKER} sources.")

    first = await GEPA(
        minibatch_size=2, max_iterations=1, use_merge=False, run_dir=run_dir
    ).optimize(
        program,
        seed,
        _trainset(),
        _marker_metric,
        Budget(max_rollouts=100),
        _harness(client),
    )
    assert (run_dir / "checkpoint.json").exists()
    first_ids = {c.id for c in first.candidates}
    assert len(first_ids) == 2  # seed + accepted child

    second = await GEPA(
        minibatch_size=2, max_iterations=5, use_merge=False, resume_from=run_dir
    ).optimize(
        program,
        seed,
        _trainset(),
        _marker_metric,
        Budget(max_rollouts=100),
        _harness(client),
    )
    second_ids = {c.id for c in second.candidates}
    # Pool is a superset with candidate ids preserved; no re-evaluated seed.
    assert first_ids <= second_ids
    assert second.scores[first.best.id] == first.scores[first.best.id]
    assert second.best.id in second_ids


# ---------------------------------------------------------------------------
# Merge path (unit-level via _attempt_merge on a hand-built diamond)
# ---------------------------------------------------------------------------


def _two_module_program() -> PromptProgram:
    sig1 = Signature(
        instruction="Draft.",
        inputs=[Field(name="question")],
        outputs=[Field(name="draft")],
    )
    sig2 = Signature(
        instruction="Finalize.",
        inputs=[Field(name="draft")],
        outputs=[Field(name="answer")],
    )
    return PromptProgram(modules=[Module("m1", sig1), Module("m2", sig2)])


def _diamond_state() -> tuple[GepaState, Candidate, Candidate, Candidate]:
    ancestor = Candidate.seed(
        modules={
            "m1": ModuleState(instruction="Draft."),
            "m2": ModuleState(instruction="Finalize."),
        }
    )
    b = ancestor.child(
        modules={
            "m1": ModuleState(instruction="Draft. B-MUT"),
            "m2": ModuleState(instruction="Finalize."),
        },
        optimizer="gepa",
    )
    c = ancestor.child(
        modules={
            "m1": ModuleState(instruction="Draft."),
            "m2": ModuleState(instruction="Finalize. C-MUT"),
        },
        optimizer="gepa",
    )
    state = GepaState()
    state.add(ancestor, [0.0, 0.0])
    state.add(b, [1.0, 0.0])  # frontier specialist on instance 0
    state.add(c, [0.0, 1.0])  # frontier specialist on instance 1
    return state, ancestor, b, c


def _merge_client() -> FakeLLMClient:
    """Each module echoes its own mutation marker when present."""

    def _respond(call: LLMCall) -> str:
        system = call.messages[0].content
        if "[[draft]]" in system:
            marker = " B-MUT" if "B-MUT" in system else ""
            return f"[[draft]]: draft{marker}"
        marker = " C-MUT" if "C-MUT" in system else ""
        return f"[[answer]]: answer{marker}"

    return FakeLLMClient(script=_respond)


async def test_merge_accepts_complementary_child() -> None:
    def metric(example: Example, prediction: Prediction) -> MetricResult:
        score = 0.5 * ("B-MUT" in prediction.outputs.get("draft", "")) + 0.5 * (
            "C-MUT" in prediction.outputs.get("answer", "")
        )
        return MetricResult(score=score)

    state, ancestor, b, c = _diamond_state()
    engine = GEPA(minibatch_size=2)
    events, emit = _collector()
    examples = [Example(inputs={"question": f"Q{i}?"}) for i in range(4)]

    accepted = await engine._attempt_merge(
        state,
        _two_module_program(),
        examples,
        examples[:2],
        metric,
        Budget(max_rollouts=100),
        _harness(_merge_client()),
        random.Random(0),
        emit,
    )

    assert accepted is True
    merged = next(cand for cid, cand in state.pool.items() if cid not in {ancestor.id, b.id, c.id})
    # Triplet rule: each mutated module comes from the parent that mutated it.
    assert merged.modules["m1"].instruction == "Draft. B-MUT"
    assert merged.modules["m2"].instruction == "Finalize. C-MUT"
    assert set(merged.parent_ids) == {b.id, c.id}
    # Merged child got a full D_pareto eval.
    assert state.scores[merged.id] == [1.0, 1.0]

    merge_events = [e for e in events if e.type == "merge_attempted"]
    assert len(merge_events) == 1
    assert merge_events[0].payload["accepted"] is True
    assert merge_events[0].payload["ancestor"] == ancestor.id
    assert set(merge_events[0].payload["parents"]) == {b.id, c.id}


async def test_merge_rejected_when_child_scores_below_parents() -> None:
    def metric(example: Example, prediction: Prediction) -> MetricResult:
        # Rewards exactly one mutation: the merged child (both) scores 0.
        n = ("B-MUT" in prediction.outputs.get("draft", "")) + (
            "C-MUT" in prediction.outputs.get("answer", "")
        )
        return MetricResult(score=1.0 if n == 1 else 0.0)

    state, ancestor, b, c = _diamond_state()
    pool_before = set(state.pool)
    events, emit = _collector()
    examples = [Example(inputs={"question": f"Q{i}?"}) for i in range(4)]

    accepted = await GEPA(minibatch_size=2)._attempt_merge(
        state,
        _two_module_program(),
        examples,
        examples[:2],
        metric,
        Budget(max_rollouts=100),
        _harness(_merge_client()),
        random.Random(0),
        emit,
    )

    assert accepted is False
    assert set(state.pool) == pool_before
    merge_events = [e for e in events if e.type == "merge_attempted"]
    assert merge_events[0].payload["accepted"] is False


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


async def test_events_emitted() -> None:
    program = _program()
    seed = _seed(program)
    events, emit = _collector()
    await GEPA(minibatch_size=2, max_iterations=3, use_merge=False).optimize(
        program,
        seed,
        _trainset(),
        _marker_metric,
        Budget(max_rollouts=100),
        _harness(_client(f"Answer the question. {MARKER} sources.")),
        emit=emit,
    )

    types = {e.type for e in events}
    assert {
        "run_started",
        "candidate_proposed",
        "minibatch_scored",
        "full_eval",
        "pareto_updated",
        "run_finished",
    } <= types
    assert events[0].type == "run_started"
    assert events[-1].type == "run_finished"


# ---------------------------------------------------------------------------
# Finding 1 — Truncated full-eval corrupts Pareto matrix
# ---------------------------------------------------------------------------


async def test_full_eval_truncated_marks_partial() -> None:
    """Budget-truncated full eval stores 0-filled vector and marks candidate as partial."""
    program = _program()
    seed = _seed(program)
    state = GepaState()
    d_pareto = _trainset(4)
    # Allow only 2 of 4 rollouts so eval is truncated
    budget = Budget(max_rollouts=2)
    events: list[RunEvent] = []
    await GEPA(minibatch_size=2)._full_eval(
        state,
        program,
        seed,
        d_pareto,
        _marker_metric,
        budget,
        _harness(_client("plain")),
        events.append,
    )
    assert seed.id in state.partial
    assert len(state.scores[seed.id]) == 4  # 0.0-filled alignment


async def test_resume_repairs_partial_vector(tmp_path: Path) -> None:
    """Resume with fresh budget re-evaluates partial candidates; clears partial flag."""
    import json

    program = _program()
    seed = _seed(program)
    trainset = _trainset(8)
    run_dir = tmp_path / "run"

    # Phase 1: budget so tight only 2 of 4 pareto examples evaluated → partial
    await GEPA(
        minibatch_size=2,
        n_pareto=4,
        max_iterations=0,
        use_merge=False,
        run_dir=run_dir,
    ).optimize(
        program,
        seed,
        trainset,
        _marker_metric,
        Budget(max_rollouts=2),
        _harness(_client("plain")),  # 0.0 scores
    )

    cp = json.loads((run_dir / "checkpoint.json").read_text())
    assert seed.id in cp.get("partial", []), "checkpoint must record partial flag"

    # Phase 2: resume with generous budget and a scoring client → repair
    result = await GEPA(
        minibatch_size=2,
        n_pareto=4,
        max_iterations=0,
        use_merge=False,
        resume_from=run_dir,
    ).optimize(
        program,
        seed,
        trainset,
        _marker_metric,
        Budget(max_rollouts=100),
        _harness(_always_marker_client()),  # 1.0 scores
    )

    assert result.scores[seed.id] == 1.0, "after repair vector reflects true mean"


# ---------------------------------------------------------------------------
# Finding 2 — budget_tick never emitted
# ---------------------------------------------------------------------------


async def test_budget_tick_emitted_monotonically() -> None:
    """budget_tick events are emitted each iteration with nondecreasing rollouts_used."""
    program = _program()
    seed = _seed(program)
    events, emit = _collector()
    await GEPA(minibatch_size=2, max_iterations=3, use_merge=False).optimize(
        program,
        seed,
        _trainset(),
        _marker_metric,
        Budget(max_rollouts=100),
        _harness(_client(f"Answer. {MARKER} sources.")),
        emit=emit,
    )
    ticks = [e for e in events if e.type == "budget_tick"]
    assert len(ticks) >= 1
    used = [e.payload["rollouts_used"] for e in ticks]
    assert used == sorted(used), "rollouts_used must be nondecreasing"


# ---------------------------------------------------------------------------
# Finding 3 — Resume seed-id guard
# ---------------------------------------------------------------------------


async def test_resume_seed_guard_identical_modules(tmp_path: Path) -> None:
    """On resume, if seed id differs but modules match a pool candidate, no re-eval."""
    program = _program()
    seed = _seed(program)
    run_dir = tmp_path / "run"
    client = _client(f"Answer. {MARKER}")

    await GEPA(minibatch_size=2, max_iterations=1, use_merge=False, run_dir=run_dir).optimize(
        program,
        seed,
        _trainset(),
        _marker_metric,
        Budget(max_rollouts=100),
        _harness(client),
    )

    # Build a new seed with the SAME modules but a new id (simulates re-init)
    new_seed = Candidate.seed(modules=dict(seed.modules))
    assert new_seed.id != seed.id

    second = await GEPA(
        minibatch_size=2, max_iterations=1, use_merge=False, resume_from=run_dir
    ).optimize(
        program,
        new_seed,
        _trainset(),
        _marker_metric,
        Budget(max_rollouts=100),
        _harness(client),
    )
    # Original seed id still in pool; no duplicate added
    assert seed.id in {c.id for c in second.candidates}
    assert new_seed.id not in {c.id for c in second.candidates}


async def test_resume_seed_guard_raises_on_mismatch(tmp_path: Path) -> None:
    """On resume, if seed modules don't match any pool candidate, raise ValueError."""
    import pytest

    program = _program()
    seed = _seed(program)
    run_dir = tmp_path / "run"

    await GEPA(minibatch_size=2, max_iterations=1, use_merge=False, run_dir=run_dir).optimize(
        program,
        seed,
        _trainset(),
        _marker_metric,
        Budget(max_rollouts=100),
        _harness(_client(f"Answer. {MARKER}")),
    )

    alien_seed = Candidate.seed(
        modules={"main": ModuleState(instruction="Completely different instruction.")}
    )
    with pytest.raises(ValueError, match="resume pool does not contain the provided seed"):
        await GEPA(
            minibatch_size=2, max_iterations=1, use_merge=False, resume_from=run_dir
        ).optimize(
            program,
            alien_seed,
            _trainset(),
            _marker_metric,
            Budget(max_rollouts=100),
            _harness(_client("plain")),
        )


# ---------------------------------------------------------------------------
# Finding 4 — Merge scheduler untested
# ---------------------------------------------------------------------------


async def test_merge_scheduler_emits_merge_attempted() -> None:
    """Scheduler calls _attempt_merge via the main loop after merge_every acceptances."""
    program = _program()
    seed = _seed(program)
    events, emit = _collector()

    await GEPA(
        minibatch_size=2,
        max_iterations=5,
        use_merge=True,
        merge_every=1,
        max_merges=2,
    ).optimize(
        program,
        seed,
        _trainset(),
        _marker_metric,
        Budget(max_rollouts=200),
        _harness(_client(f"Answer. {MARKER} sources.")),
        emit=emit,
    )

    # merge_attempted must be emitted by the scheduler (not a direct call)
    merge_events = [e for e in events if e.type == "merge_attempted"]
    assert len(merge_events) >= 1
