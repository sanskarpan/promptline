"""Tests for the Promptline TUI cockpit and its event feed."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widgets import ProgressBar, RichLog, Sparkline, Static, Tree

from promptline.optimizers.base import RunEvent
from promptline.tui.app import PromptlineTUI, summarize_payload
from promptline.tui.events import RunEventFeed

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CAND_A = "aaaaaaaa1111"
CAND_B = "bbbbbbbb2222"
CAND_C = "cccccccc3333"
SEED = "seedseed0000"


def _synthetic_events() -> list[RunEvent]:
    """10 events covering the full cockpit surface."""
    return [
        RunEvent.now("run_started", optimizer="gepa"),
        RunEvent.now("candidate_proposed", candidate_id=CAND_A, parent_id=SEED),
        RunEvent.now("candidate_proposed", candidate_id=CAND_B, parents=[CAND_A]),
        RunEvent.now("candidate_proposed", candidate_id=CAND_C, parent_ids=[CAND_A]),
        RunEvent.now("full_eval", candidate_id=CAND_A, mean_score=0.5),
        RunEvent.now("full_eval", candidate_id=CAND_B, mean_score=0.75),
        RunEvent.now("budget_tick", rollouts_used=10, cost_used=0.01, max_rollouts=100),
        RunEvent.now("budget_tick", rollouts_used=42, cost_used=0.1234, max_rollouts=100),
        RunEvent.now("minibatch_scored", score=0.9, trial=1),
        RunEvent.now("run_finished", optimizer="gepa", best_id=CAND_B),
    ]


def _write_events(path: Path, events: list[RunEvent]) -> None:
    with path.open("a") as fh:
        for event in events:
            fh.write(event.model_dump_json() + "\n")


def _tree_size(tree: Tree) -> int:
    count = 0
    stack = [tree.root]
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(node.children)
    return count


# ---------------------------------------------------------------------------
# RunEventFeed.from_file
# ---------------------------------------------------------------------------


async def test_feed_from_static_file_yields_all_and_stops(tmp_path: Path) -> None:
    events = _synthetic_events()
    path = tmp_path / "events.jsonl"
    _write_events(path, events)

    feed = RunEventFeed.from_file(path, follow=True, poll_interval=0.01)
    seen = [event async for event in feed]

    assert len(seen) == len(events)
    assert [e.type for e in seen] == [e.type for e in events]
    assert seen[-1].type == "run_finished"


async def test_feed_from_file_no_follow_stops_at_eof(tmp_path: Path) -> None:
    events = _synthetic_events()[:3]  # no run_finished
    path = tmp_path / "events.jsonl"
    _write_events(path, events)

    feed = RunEventFeed.from_file(path, follow=False)
    seen = [event async for event in feed]
    assert len(seen) == 3


async def test_feed_from_growing_file(tmp_path: Path) -> None:
    events = _synthetic_events()
    path = tmp_path / "events.jsonl"
    _write_events(path, events[:4])

    feed = RunEventFeed.from_file(path, follow=True, poll_interval=0.01)
    seen: list[RunEvent] = []

    async def _collect() -> None:
        async for event in feed:
            seen.append(event)

    task = asyncio.create_task(_collect())
    while len(seen) < 4:
        await asyncio.sleep(0.01)
    _write_events(path, events[4:])  # append the rest, incl. run_finished
    await asyncio.wait_for(task, timeout=5.0)

    assert len(seen) == len(events)
    assert seen[-1].type == "run_finished"


async def _drain(feed: RunEventFeed) -> list[RunEvent]:
    return [event async for event in feed]


async def test_feed_idle_timeout_stops(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_events(path, _synthetic_events()[:2])
    feed = RunEventFeed.from_file(path, follow=True, poll_interval=0.01, idle_timeout=0.05)
    seen = await asyncio.wait_for(_drain(feed), timeout=5.0)
    assert len(seen) == 2


# ---------------------------------------------------------------------------
# PromptlineTUI
# ---------------------------------------------------------------------------


async def test_app_mounts_all_panes() -> None:
    app = PromptlineTUI(feed=RunEventFeed.from_events([]), run_id="testrun")
    async with app.run_test() as pilot:
        assert app.query_one("#score-pane")
        assert app.query_one("#budget-pane")
        assert app.query_one("#lineage-pane")
        assert app.query_one("#events-pane")
        assert app.query_one("#score-spark", Sparkline)
        assert app.query_one("#budget-bar", ProgressBar)
        assert app.query_one("#lineage", Tree)
        assert app.query_one("#events", RichLog)
        # Pane titles are uppercase.
        titles = {str(static.content) for static in app.query(".title").results(Static)}
        assert {"SCORE", "BUDGET", "LINEAGE", "EVENTS"} <= titles
        await pilot.pause()


async def test_app_consumes_synthetic_feed() -> None:
    events = _synthetic_events()
    app = PromptlineTUI(feed=RunEventFeed.from_events(events), run_id="testrun")
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        # Lineage: root (seed) + 3 proposed candidates.
        tree = app.query_one("#lineage", Tree)
        assert _tree_size(tree) == 4
        # Parent edges resolved: B and C hang off A.
        node_a = app._tree_nodes[CAND_A]
        assert {app._tree_nodes[CAND_B], app._tree_nodes[CAND_C]} == set(node_a.children)
        # Scored nodes carry their full-eval score in the label.
        assert f"{CAND_B[:8]} 0.750" in str(node_a.children[0].label) or any(
            "0.750" in str(child.label) for child in node_a.children
        )

        # Budget reflects the LAST budget_tick.
        bar = app.query_one("#budget-bar", ProgressBar)
        assert bar.total == 100
        assert bar.progress == 42
        assert app.cost_used == 0.1234
        assert "$0.1234" in str(app.query_one("#cost-line", Static).content)

        # Status shows FINISHED in the header.
        assert app.status == "FINISHED"
        assert "FINISHED" in str(app.query_one("#header", Static).content)

        # Event log has one line per event (>= 10).
        log = app.query_one("#events", RichLog)
        assert len(log.lines) >= 10


async def test_sparkline_collects_full_eval_means() -> None:
    events = _synthetic_events()
    app = PromptlineTUI(feed=RunEventFeed.from_events(events), run_id="testrun")
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        expected = [e.payload["mean_score"] for e in events if e.type == "full_eval"]
        assert app.full_eval_means == expected
        spark = app.query_one("#score-spark", Sparkline)
        assert list(spark.data) == expected
        # Best-so-far figure shows the max full-eval mean.
        assert app.best_score == max(expected)
        assert "0.750" in str(app.query_one("#best-score", Static).content)


async def test_f_toggles_follow_and_q_quits() -> None:
    app = PromptlineTUI(feed=RunEventFeed.from_events([]), run_id="testrun")
    async with app.run_test() as pilot:
        assert app.following is True
        await pilot.press("f")
        assert app.following is False
        await pilot.press("f")
        assert app.following is True
        await pilot.press("q")
    assert app._exit


# ---------------------------------------------------------------------------
# Status-transition edge cases
# ---------------------------------------------------------------------------


async def test_feed_ends_without_run_finished_sets_failed() -> None:
    """If the feed is exhausted while status == RUNNING, TUI must show FAILED."""
    events = [
        RunEvent.now("run_started", optimizer="gepa"),
        RunEvent.now("candidate_proposed", candidate_id=CAND_A),
        # No run_finished — simulates an optimizer crash / OOM-kill.
    ]
    app = PromptlineTUI(feed=RunEventFeed.from_events(events), run_id="testrun")
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.status == "FAILED"
        assert "FAILED" in str(app.query_one("#header", Static).content)


async def test_malformed_line_does_not_flip_failed(tmp_path: Path) -> None:
    """A partially-flushed / corrupt JSONL line must be skipped, not crash the TUI."""
    path = tmp_path / "events.jsonl"
    events = _synthetic_events()
    with path.open("w") as fh:
        fh.write(events[0].model_dump_json() + "\n")  # run_started
        fh.write("{broken json\n")  # malformed line
        for event in events[1:]:
            fh.write(event.model_dump_json() + "\n")

    feed = RunEventFeed.from_file(path, follow=False)
    seen = [event async for event in feed]

    # Malformed line skipped; all valid events present.
    assert len(seen) == len(events)
    assert seen[-1].type == "run_finished"


# ---------------------------------------------------------------------------
# Payload summary formatting
# ---------------------------------------------------------------------------


def test_summarize_payload_formats() -> None:
    summary = summarize_payload(
        {"mean_score": 0.123456, "cost_used": 0.98765, "candidate_id": "abc"}
    )
    assert "mean_score=0.123" in summary
    assert "cost_used=0.9877" in summary
    assert "candidate_id=abc" in summary
