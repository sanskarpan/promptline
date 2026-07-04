"""Async event feeds for the Promptline TUI.

A :class:`RunEventFeed` is an async iterator over :class:`RunEvent` objects
sourced from a run's ``events.jsonl`` (optionally tailed while it grows), an
SSE endpoint (``GET /runs/{id}/events``), or an in-memory list (tests).
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

from promptline.optimizers.base import RunEvent


def _parse_line(line: str) -> RunEvent | None:
    stripped = line.strip()
    if not stripped:
        return None
    return RunEvent.model_validate_json(stripped)


class RunEventFeed:
    """An async iterator over optimizer run events.

    Construct via :meth:`from_file`, :meth:`from_url` or :meth:`from_events`.
    Iteration ends after a ``run_finished`` event, when the underlying source
    is exhausted (non-follow mode), or when *idle_timeout* elapses with no new
    data (follow mode; ``None`` waits forever).
    """

    def __init__(self, source: AsyncIterator[RunEvent]) -> None:
        self._source = source

    def __aiter__(self) -> AsyncIterator[RunEvent]:
        return self._source.__aiter__()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_events(cls, events: Iterable[RunEvent]) -> RunEventFeed:
        """Feed from an in-memory sequence (used by tests)."""
        snapshot = list(events)

        async def _gen() -> AsyncIterator[RunEvent]:
            for event in snapshot:
                yield event

        return cls(_gen())

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        follow: bool = True,
        poll_interval: float = 0.2,
        idle_timeout: float | None = None,
    ) -> RunEventFeed:
        """Feed from an ``events.jsonl`` file, tailing it while it grows.

        Parameters
        ----------
        path:
            Path to the JSONL event log (may not exist yet in follow mode).
        follow:
            Keep polling for appended lines after reaching EOF.
        poll_interval:
            Seconds between tail polls while following.
        idle_timeout:
            Stop after this many seconds without new data (follow mode only);
            ``None`` follows forever (until ``run_finished``).
        """
        file_path = Path(path)

        async def _gen() -> AsyncIterator[RunEvent]:
            offset = 0
            partial = ""
            idle = 0.0
            while True:
                chunk = ""
                if file_path.exists():
                    with file_path.open("r") as fh:
                        fh.seek(offset)
                        chunk = fh.read()
                        offset = fh.tell()
                if chunk:
                    idle = 0.0
                    partial += chunk
                    lines = partial.split("\n")
                    partial = lines.pop()  # trailing incomplete line (if any)
                    for line in lines:
                        try:
                            event = _parse_line(line)
                        except Exception:
                            # Skip malformed / partially-flushed lines and keep
                            # tailing — a transient write-flush artefact must not
                            # permanently flip TUI status to FAILED.
                            continue
                        if event is None:
                            continue
                        yield event
                        if event.type == "run_finished":
                            return
                if not follow:
                    return
                await asyncio.sleep(poll_interval)
                idle += poll_interval
                if idle_timeout is not None and idle >= idle_timeout:
                    return

        return cls(_gen())

    @classmethod
    def from_url(cls, url: str) -> RunEventFeed:
        """Feed from an SSE endpoint — minimal ``data:`` line parser.

        Network path; kept thin and untested by default.
        """

        async def _gen() -> AsyncIterator[RunEvent]:
            import httpx

            async with httpx.AsyncClient(timeout=None) as client:  # pragma: no cover
                async with client.stream("GET", url) as response:
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        event = _parse_line(line[len("data:"):])
                        if event is None:
                            continue
                        yield event
                        if event.type == "run_finished":
                            return

        return cls(_gen())
