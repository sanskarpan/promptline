from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from promptline.core.program import PromptProgram
from promptline.core.types import Candidate, Example
from promptline.eval.harness import Budget, EvalHarness, Metric

# ---------------------------------------------------------------------------
# Run event
# ---------------------------------------------------------------------------

_EventType = Literal[
    "run_started",
    "candidate_proposed",
    "minibatch_scored",
    "full_eval",
    "pareto_updated",
    "merge_attempted",
    "budget_tick",
    "run_finished",
]


class RunEvent(BaseModel):
    """A structured event emitted during an optimizer run.

    Use :meth:`RunEvent.now` to create events with the current timestamp.
    """

    type: _EventType
    payload: dict = Field(default_factory=dict)
    ts: float = 0.0

    @classmethod
    def now(cls, type: str, **payload: object) -> RunEvent:  # noqa: A002
        """Create a :class:`RunEvent` stamped with the current wall time."""
        return cls(type=type, payload=dict(payload), ts=time.time())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Optimize result
# ---------------------------------------------------------------------------


class OptimizeResult(BaseModel):
    """Output of a completed optimizer run."""

    best: Candidate
    candidates: list[Candidate]
    scores: dict[str, float]
    events_count: int = 0


# ---------------------------------------------------------------------------
# Abstract optimizer
# ---------------------------------------------------------------------------


class Optimizer(ABC):
    """Base class for all Promptline optimizers.

    Subclasses must set a ``name`` class/instance attribute and implement
    :meth:`optimize`.
    """

    name: str

    @abstractmethod
    async def optimize(
        self,
        program: PromptProgram,
        seed: Candidate,
        trainset: list[Example],
        metric: Metric,
        budget: Budget,
        harness: EvalHarness,
        emit: Callable[[RunEvent], None] = lambda e: None,
    ) -> OptimizeResult:
        """Run the optimization loop and return the best candidate found.

        Parameters
        ----------
        program:
            The program structure (modules + signatures) being optimized.
        seed:
            Starting candidate (e.g. the hand-written baseline).
        trainset:
            Training examples used for scoring candidates.
        metric:
            Scoring function — see :data:`~promptline.eval.harness.Metric`.
        budget:
            Hard limit on rollouts and/or cost.
        harness:
            Evaluation harness to call for scoring.
        emit:
            Optional callback for progress events; defaults to a no-op.
        """


# ---------------------------------------------------------------------------
# Run recorder
# ---------------------------------------------------------------------------


class RunRecorder:
    """Persists optimizer run events and checkpoints to disk.

    Parameters
    ----------
    run_dir:
        Directory where ``events.jsonl`` and ``checkpoint.json`` are stored.
        Created automatically on first write.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._count: int = 0

    @property
    def _events_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def _checkpoint_path(self) -> Path:
        return self.run_dir / "checkpoint.json"

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------

    def emit(self, event: RunEvent) -> None:
        """Append *event* to the JSONL log, flushing immediately."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self._events_path.open("a") as fh:
            fh.write(event.model_dump_json() + "\n")
            fh.flush()
        self._count += 1

    @property
    def count(self) -> int:
        """Number of events emitted via :meth:`emit` in this session."""
        return self._count

    @staticmethod
    def read_events(run_dir: Path) -> list[RunEvent]:
        """Read all events from *run_dir/events.jsonl* in order."""
        events_path = run_dir / "events.jsonl"
        if not events_path.exists():
            return []
        events: list[RunEvent] = []
        for line in events_path.read_text().splitlines():
            stripped = line.strip()
            if stripped:
                events.append(RunEvent.model_validate_json(stripped))
        return events

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, state: dict) -> None:
        """Atomically write *state* to ``checkpoint.json``.

        Writes to a sibling ``.tmp`` file first, then uses :func:`os.replace`
        so a concurrent reader never sees a partially-written file.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._checkpoint_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state))
        os.replace(tmp_path, self._checkpoint_path)

    def load_checkpoint(self) -> dict:
        """Load the last saved checkpoint, or return an empty dict."""
        if not self._checkpoint_path.exists():
            return {}
        return json.loads(self._checkpoint_path.read_text())
