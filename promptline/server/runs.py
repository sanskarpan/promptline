"""Async run management for the Promptline server.

:class:`RunManager` starts optimizer runs as :mod:`asyncio` tasks.  Each run
gets a :class:`~promptline.optimizers.base.RunRecorder` rooted at
``base_dir/<run_id>``; the recorder's ``emit`` is handed to the coroutine
factory so optimizer events land in ``events.jsonl`` where the SSE endpoint
can replay/tail them.
"""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path

from promptline.optimizers.base import RunEvent, RunRecorder

#: A factory receiving the recorder's ``emit`` and returning the run coroutine.
CoroFactory = Callable[[Callable[[RunEvent], None]], Coroutine]


@dataclass
class RunInfo:
    """Bookkeeping for one managed run."""

    run_id: str
    status: str = "running"  # "running" | "finished" | "failed"
    summary: dict = field(default_factory=dict)
    error: str = ""
    task: asyncio.Task | None = None

    def public(self) -> dict:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "summary": self.summary,
            "error": self.error,
        }


def _summarize(result: object) -> dict:
    """Best-effort summary of an optimizer result (best id/score)."""
    best = getattr(result, "best", None)
    if best is not None:
        scores = getattr(result, "scores", None) or {}
        return {"best_id": best.id, "best_score": scores.get(best.id)}
    if isinstance(result, dict):
        return result
    return {}


class RunManager:
    """Starts and tracks optimizer runs as asyncio tasks under *base_dir*."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self._runs: dict[str, RunInfo] = {}

    def events_path(self, run_id: str) -> Path:
        return self.base_dir / run_id / "events.jsonl"

    def start(self, coro_factory: CoroFactory, run_id: str | None = None) -> str:
        """Launch a run and return its id (uuid4 hex when not supplied)."""
        run_id = run_id or uuid.uuid4().hex
        recorder = RunRecorder(self.base_dir / run_id)
        info = RunInfo(run_id=run_id)
        self._runs[run_id] = info
        info.task = asyncio.create_task(self._run(info, coro_factory(recorder.emit)))
        return run_id

    async def _run(self, info: RunInfo, coro: Coroutine) -> None:
        try:
            result = await coro
        except Exception as exc:  # noqa: BLE001 — surfaced via the API
            info.status = "failed"
            info.error = f"{type(exc).__name__}: {exc}"
            return
        info.summary = _summarize(result)
        info.status = "finished"

    def get(self, run_id: str) -> dict | None:
        info = self._runs.get(run_id)
        return info.public() if info is not None else None

    def list(self) -> list[dict]:
        return [info.public() for info in self._runs.values()]
