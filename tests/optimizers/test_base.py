from __future__ import annotations

import time
from pathlib import Path

import pytest

from promptline.core.llm import FakeLLMClient
from promptline.core.program import ModelConfig, PromptProgram
from promptline.core.types import Candidate, Example, ModuleState
from promptline.eval.harness import Budget, EvalHarness, MetricResult
from promptline.optimizers.base import (
    OptimizeResult,
    Optimizer,
    RunEvent,
    RunRecorder,
)

# ---------------------------------------------------------------------------
# RunEvent
# ---------------------------------------------------------------------------


def test_run_event_now_sets_ts() -> None:
    before = time.time()
    event = RunEvent.now("run_started")
    after = time.time()
    assert before <= event.ts <= after


def test_run_event_now_stores_payload() -> None:
    event = RunEvent.now("candidate_proposed", score=0.85, step=3)
    assert event.payload == {"score": 0.85, "step": 3}
    assert event.type == "candidate_proposed"


def test_run_event_now_empty_payload() -> None:
    event = RunEvent.now("run_finished")
    assert event.payload == {}


def test_run_event_default_ts_zero() -> None:
    event = RunEvent(type="run_started")
    assert event.ts == 0.0


def test_run_event_all_literal_types_accepted() -> None:
    valid_types = [
        "run_started",
        "candidate_proposed",
        "minibatch_scored",
        "full_eval",
        "pareto_updated",
        "merge_attempted",
        "budget_tick",
        "run_finished",
    ]
    for t in valid_types:
        e = RunEvent(type=t)  # type: ignore[arg-type]
        assert e.type == t


# ---------------------------------------------------------------------------
# OptimizeResult
# ---------------------------------------------------------------------------


def test_optimize_result_defaults() -> None:
    cand = Candidate.seed(modules={"main": ModuleState(instruction="test")})
    result = OptimizeResult(best=cand, candidates=[cand], scores={"main": 0.9})
    assert result.events_count == 0


# ---------------------------------------------------------------------------
# Optimizer ABC
# ---------------------------------------------------------------------------


def test_optimizer_abc_cannot_instantiate() -> None:
    with pytest.raises(TypeError):
        Optimizer()  # type: ignore[abstract]


def test_optimizer_abc_requires_optimize_implementation() -> None:
    """A concrete subclass that doesn't implement optimize() must also fail."""
    class IncompleteOptimizer(Optimizer):
        name = "incomplete"

    with pytest.raises(TypeError):
        IncompleteOptimizer()


def test_optimizer_concrete_subclass_instantiates() -> None:
    class ConcreteOptimizer(Optimizer):
        name = "concrete"

        async def optimize(self, program, seed, trainset, metric, budget, harness, emit=lambda e: None):
            return OptimizeResult(best=seed, candidates=[seed], scores={})

    opt = ConcreteOptimizer()
    assert opt.name == "concrete"


# ---------------------------------------------------------------------------
# RunRecorder — event round-trip
# ---------------------------------------------------------------------------


def test_recorder_emit_and_read_round_trip(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    recorder = RunRecorder(run_dir)

    event_types = [
        "run_started",
        "candidate_proposed",
        "minibatch_scored",
        "full_eval",
        "run_finished",
    ]
    sent: list[RunEvent] = []
    for t in event_types:
        e = RunEvent.now(t, step=event_types.index(t))
        recorder.emit(e)
        sent.append(e)

    assert recorder.count == 5
    read = RunRecorder.read_events(run_dir)
    assert len(read) == 5
    for original, loaded in zip(sent, read):
        assert loaded.type == original.type
        assert loaded.payload == original.payload
        assert loaded.ts == pytest.approx(original.ts, abs=1e-6)


def test_recorder_read_events_missing_dir_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    assert RunRecorder.read_events(missing) == []


def test_recorder_creates_run_dir_on_first_emit(tmp_path: Path) -> None:
    run_dir = tmp_path / "nested" / "run"
    recorder = RunRecorder(run_dir)
    assert not run_dir.exists()
    recorder.emit(RunEvent(type="run_started"))
    assert run_dir.exists()
    assert (run_dir / "events.jsonl").exists()


def test_recorder_count_increments(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path / "run")
    assert recorder.count == 0
    recorder.emit(RunEvent(type="run_started"))
    recorder.emit(RunEvent(type="run_finished"))
    assert recorder.count == 2


# ---------------------------------------------------------------------------
# RunRecorder — checkpoint round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    run_dir = tmp_path / "run2"
    recorder = RunRecorder(run_dir)
    state = {
        "epoch": 5,
        "best_score": 0.85,
        "candidates": ["abc", "def"],
        "nested": {"key": True},
    }
    recorder.save_checkpoint(state)
    loaded = recorder.load_checkpoint()
    assert loaded == state


def test_checkpoint_load_missing_returns_empty(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path / "run3")
    assert recorder.load_checkpoint() == {}


def test_checkpoint_overwrite(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path / "run4")
    recorder.save_checkpoint({"v": 1})
    recorder.save_checkpoint({"v": 2})
    assert recorder.load_checkpoint() == {"v": 2}
