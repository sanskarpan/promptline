"""Promptline TUI cockpit — a live view over an optimizer run's event stream.

Four flat panes in the opencode/Hermes terminal aesthetic: SCORE (best-so-far
plus a sparkline of full-eval means), BUDGET (rollouts progress + cost),
LINEAGE (candidate tree) and EVENTS (raw event log).
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import ProgressBar, RichLog, Sparkline, Static, Tree
from textual.widgets.tree import TreeNode

from promptline.optimizers.base import RunEvent
from promptline.tui.events import RunEventFeed

ACCENT = "#4af6c3"
RED = "#ff5f56"
DIM = "#666666"

#: Event type -> log tag color.
_TYPE_COLORS: dict[str, str] = {
    "run_started": ACCENT,
    "run_finished": DIM,
    "full_eval": ACCENT,
    "candidate_proposed": "#8be9fd",
    "minibatch_scored": "#bd93f9",
    "pareto_updated": "#f1fa8c",
    "merge_attempted": "#f1fa8c",
    "budget_tick": DIM,
}


def _fmt_value(key: str, value: object) -> str:
    """Compact payload value: scores 3 decimals, costs 4, strings truncated."""
    if isinstance(value, float):
        if "cost" in key:
            return f"{value:.4f}"
        if "score" in key or "mean" in key:
            return f"{value:.3f}"
        return f"{value:g}"
    text = str(value)
    if len(text) > 40:
        text = text[:37] + "..."
    return text.replace("\n", " ")


def summarize_payload(payload: dict) -> str:
    """One-line ``k=v`` summary of an event payload."""
    return " ".join(f"{k}={_fmt_value(k, v)}" for k, v in payload.items())


def _parents_of(payload: dict) -> list[str]:
    """Extract parent candidate ids from either payload shape."""
    parents = payload.get("parents") or payload.get("parent_ids")
    if parents:
        return [str(p) for p in parents]
    parent_id = payload.get("parent_id")
    return [str(parent_id)] if parent_id else []


class PromptlineTUI(App):
    """Live cockpit over a :class:`RunEventFeed`."""

    TITLE = "PROMPTLINE"

    CSS = f"""
    Screen {{
        background: #0a0a0a;
        color: #d0d0d0;
    }}
    #header {{
        height: 1;
        background: #111111;
        color: {ACCENT};
        padding: 0 1;
    }}
    .panel {{
        border: solid #2a2a2a;
        background: #111111;
    }}
    .title {{
        height: 1;
        padding: 0 1;
        color: {ACCENT};
        text-style: bold;
    }}
    #body {{
        height: 1fr;
    }}
    #left {{
        width: 44;
    }}
    #score-pane {{
        height: 1fr;
    }}
    #best-score {{
        height: 3;
        content-align: center middle;
        text-align: center;
        color: {ACCENT};
        text-style: bold;
    }}
    #score-spark {{
        height: 3;
        margin: 0 1;
    }}
    Sparkline > .sparkline--max-color {{
        color: {ACCENT};
    }}
    Sparkline > .sparkline--min-color {{
        color: #1f6c55;
    }}
    #budget-pane {{
        height: 7;
    }}
    #budget-bar {{
        margin: 0 1;
    }}
    Bar > .bar--bar {{
        color: {ACCENT};
        background: #2a2a2a;
    }}
    Bar > .bar--complete {{
        color: {ACCENT};
    }}
    #cost-line {{
        padding: 0 1;
        color: #d0d0d0;
    }}
    #lineage-pane {{
        width: 1fr;
    }}
    #lineage {{
        background: #111111;
    }}
    #events-pane {{
        height: 14;
    }}
    #events {{
        background: #111111;
        scrollbar-color: #2a2a2a;
    }}
    """

    BINDINGS = [
        ("q", "quit", "QUIT"),
        ("f", "toggle_follow", "FOLLOW"),
    ]

    def __init__(self, feed: RunEventFeed, run_id: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.feed = feed
        self.run_id = run_id
        self.optimizer = ""
        self.status = "WAITING"
        self.best_score: float | None = None
        #: Full-eval mean scores in arrival order (drives the sparkline).
        self.full_eval_means: list[float] = []
        self.rollouts_used = 0
        self.max_rollouts: int | None = None
        self.cost_used = 0.0
        self.following = True
        self._resume = asyncio.Event()
        self._resume.set()
        self._tree_nodes: dict[str, TreeNode] = {}
        self._candidate_scores: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static("", id="header")
        with Horizontal(id="body"):
            with Vertical(id="left"):
                with Vertical(id="score-pane", classes="panel"):
                    yield Static("SCORE", classes="title")
                    yield Static("--", id="best-score")
                    yield Sparkline([], id="score-spark")
                with Vertical(id="budget-pane", classes="panel"):
                    yield Static("BUDGET", classes="title")
                    yield ProgressBar(id="budget-bar", show_eta=False)
                    yield Static("$0.0000", id="cost-line")
            with Vertical(id="lineage-pane", classes="panel"):
                yield Static("LINEAGE", classes="title")
                yield Tree("SEED", id="lineage")
        with Vertical(id="events-pane", classes="panel"):
            yield Static("EVENTS", classes="title")
            yield RichLog(id="events", wrap=False, max_lines=1000)

    def on_mount(self) -> None:
        self.query_one("#lineage", Tree).root.expand()
        self._refresh_header()
        self.run_worker(self._consume(), exclusive=True)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_toggle_follow(self) -> None:
        self.following = not self.following
        if self.following:
            self._resume.set()
        else:
            self._resume.clear()
        self._refresh_header()

    # ------------------------------------------------------------------
    # Event consumption
    # ------------------------------------------------------------------

    async def _consume(self) -> None:
        try:
            async for event in self.feed:
                await self._resume.wait()
                self.handle_event(event)
        except Exception:
            self.status = "FAILED"
            self._refresh_header()
            raise
        if self.status == "RUNNING":
            # Feed ended without a run_finished event.
            self.status = "FAILED"
            self._refresh_header()

    def handle_event(self, event: RunEvent) -> None:
        payload = event.payload
        if event.type == "run_started":
            self.optimizer = str(payload.get("optimizer", ""))
            self.status = "RUNNING"
        elif event.type == "run_finished":
            self.status = "FINISHED"
        elif event.type == "candidate_proposed":
            self._on_candidate_proposed(payload)
        elif event.type == "full_eval":
            self._on_full_eval(payload)
        elif event.type == "budget_tick":
            self._on_budget_tick(payload)
        self._log_event(event)
        self._refresh_header()

    # ------------------------------------------------------------------
    # Pane updates
    # ------------------------------------------------------------------

    def _refresh_header(self) -> None:
        status_style = {
            "RUNNING": ACCENT,
            "FINISHED": DIM,
            "FAILED": RED,
        }.get(self.status, DIM)
        text = Text()
        text.append("PROMPTLINE ", style=f"bold {ACCENT}")
        text.append(self.run_id[:12] or "-", style="#d0d0d0")
        text.append("  ")
        text.append((self.optimizer or "-").upper(), style="#d0d0d0")
        text.append("  ")
        text.append(self.status, style=f"bold {status_style}")
        if not self.following:
            text.append("  PAUSED", style=f"bold {RED}")
        self.query_one("#header", Static).update(text)

    def _on_full_eval(self, payload: dict) -> None:
        mean = payload.get("mean_score")
        if not isinstance(mean, int | float):
            return
        mean = float(mean)
        self.full_eval_means.append(mean)
        if self.best_score is None or mean > self.best_score:
            self.best_score = mean
        self.query_one("#best-score", Static).update(f"{self.best_score:.3f}")
        self.query_one("#score-spark", Sparkline).data = list(self.full_eval_means)
        candidate_id = payload.get("candidate_id")
        if candidate_id:
            self._candidate_scores[str(candidate_id)] = mean
            node = self._tree_nodes.get(str(candidate_id))
            if node is not None:
                node.set_label(self._node_label(str(candidate_id)))

    def _node_label(self, candidate_id: str) -> str:
        short = candidate_id[:8]
        score = self._candidate_scores.get(candidate_id)
        return f"{short} {score:.3f}" if score is not None else short

    def _on_candidate_proposed(self, payload: dict) -> None:
        candidate_id = payload.get("candidate_id")
        if not candidate_id:
            # Some optimizers (mipro proposals) emit no candidate id.
            return
        candidate_id = str(candidate_id)
        if candidate_id in self._tree_nodes:
            return
        tree = self.query_one("#lineage", Tree)
        parent_node = tree.root
        for parent_id in _parents_of(payload):
            known = self._tree_nodes.get(parent_id)
            if known is not None:
                parent_node = known
                break
        node = parent_node.add(self._node_label(candidate_id), expand=True)
        self._tree_nodes[candidate_id] = node

    def _on_budget_tick(self, payload: dict) -> None:
        rollouts = payload.get("rollouts_used")
        if isinstance(rollouts, int | float):
            self.rollouts_used = int(rollouts)
        max_rollouts = payload.get("max_rollouts")
        if isinstance(max_rollouts, int | float):
            self.max_rollouts = int(max_rollouts)
        cost = payload.get("cost_used")
        if isinstance(cost, int | float):
            self.cost_used = float(cost)

        bar = self.query_one("#budget-bar", ProgressBar)
        if self.max_rollouts:
            bar.update(total=self.max_rollouts, progress=self.rollouts_used)
        cost_line = f"${self.cost_used:.4f}"
        max_cost = payload.get("max_cost_usd")
        if isinstance(max_cost, int | float):
            cost_line += f" / ${float(max_cost):.4f}"
        cost_line += f"  ROLLOUTS {self.rollouts_used}"
        if self.max_rollouts:
            cost_line += f"/{self.max_rollouts}"
        self.query_one("#cost-line", Static).update(cost_line)

    def _log_event(self, event: RunEvent) -> None:
        log = self.query_one("#events", RichLog)
        stamp = datetime.fromtimestamp(event.ts, tz=UTC).strftime("%H:%M:%S")
        line = Text()
        line.append(stamp, style=DIM)
        line.append(" ")
        line.append(
            f"{event.type:<18}", style=f"bold {_TYPE_COLORS.get(event.type, DIM)}"
        )
        line.append(" ")
        line.append(summarize_payload(event.payload), style="#a0a0a0")
        log.write(line)
