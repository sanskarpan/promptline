"""SQLite-backed prompt registry.

Stores registered :class:`~promptline.core.types.Candidate` prompts, their
eval history, and a per-program *active* pointer.  :meth:`PromptRegistry.activate`
is the ONLY method that moves the pointer; every pointer move (activate or
rollback) is appended to ``activation_history``.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from promptline.core.types import Candidate

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    id TEXT PRIMARY KEY,
    program TEXT NOT NULL,
    candidate_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    run_id TEXT NOT NULL DEFAULT '',
    parent_ids_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS evals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id TEXT NOT NULL,
    dataset_hash TEXT NOT NULL,
    mean_score REAL NOT NULL,
    n INTEGER NOT NULL,
    report_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS active (
    program TEXT PRIMARY KEY,
    prompt_id TEXT NOT NULL,
    activated_at TEXT NOT NULL,
    gate_report_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS activation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    program TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    activated_at TEXT NOT NULL,
    action TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PromptRegistry:
    """Thread-safe (``threading.Lock``) prompt store at ``<path>/registry.db``."""

    def __init__(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.root = path
        self._path = path / "registry.db"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def register(self, candidate: Candidate, program: str, run_id: str = "") -> str:
        """Store *candidate* under *program*.  Idempotent on the same id."""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO prompts "
                "(id, program, candidate_json, created_at, run_id, parent_ids_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    candidate.id,
                    program,
                    candidate.model_dump_json(),
                    _now(),
                    run_id,
                    json.dumps(candidate.parent_ids),
                ),
            )
            self._conn.commit()
        return candidate.id

    def get(self, prompt_id: str) -> Candidate | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT candidate_json FROM prompts WHERE id = ?", (prompt_id,)
            ).fetchone()
        if row is None:
            return None
        return Candidate.model_validate_json(row[0])

    def list_prompts(self, program: str) -> list[dict]:
        """All prompts for *program* with the mean score of their latest eval."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT p.id, p.created_at, p.run_id, "
                "  (SELECT e.mean_score FROM evals e WHERE e.prompt_id = p.id "
                "   ORDER BY e.id DESC LIMIT 1) "
                "FROM prompts p WHERE p.program = ? ORDER BY p.rowid",
                (program,),
            ).fetchall()
        return [
            {"id": r[0], "created_at": r[1], "run_id": r[2], "mean_score": r[3]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Evals (append-only)
    # ------------------------------------------------------------------

    def record_eval(
        self,
        prompt_id: str,
        dataset_hash: str,
        mean_score: float,
        n: int,
        report_json: str = "{}",
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO evals "
                "(prompt_id, dataset_hash, mean_score, n, report_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (prompt_id, dataset_hash, mean_score, n, report_json, _now()),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Active pointer
    # ------------------------------------------------------------------

    def get_active(self, program: str) -> tuple[str, Candidate] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT a.prompt_id, p.candidate_json FROM active a "
                "JOIN prompts p ON p.id = a.prompt_id WHERE a.program = ?",
                (program,),
            ).fetchone()
        if row is None:
            return None
        return row[0], Candidate.model_validate_json(row[1])

    def get_active_info(self, program: str) -> dict | None:
        """Active pointer details: prompt_id, activated_at and the candidate."""
        with self._lock:
            row = self._conn.execute(
                "SELECT a.prompt_id, a.activated_at, p.candidate_json "
                "FROM active a JOIN prompts p ON p.id = a.prompt_id "
                "WHERE a.program = ?",
                (program,),
            ).fetchone()
        if row is None:
            return None
        return {
            "prompt_id": row[0],
            "activated_at": row[1],
            "candidate": Candidate.model_validate_json(row[2]),
        }

    def activate(
        self, program: str, prompt_id: str, gate_report_json: str = "{}"
    ) -> None:
        """Move the active pointer for *program* to *prompt_id*.

        This is the only method that moves the pointer forward.  Raises
        :class:`KeyError` when the prompt is not registered.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM prompts WHERE id = ?", (prompt_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"prompt {prompt_id!r} is not registered")
            ts = _now()
            self._conn.execute(
                "INSERT OR REPLACE INTO active "
                "(program, prompt_id, activated_at, gate_report_json) "
                "VALUES (?, ?, ?, ?)",
                (program, prompt_id, ts, gate_report_json),
            )
            self._conn.execute(
                "INSERT INTO activation_history "
                "(program, prompt_id, activated_at, action) VALUES (?, ?, ?, ?)",
                (program, prompt_id, ts, "activate"),
            )
            self._conn.commit()

    def rollback(self, program: str) -> str:
        """Revert *program* to the previous distinct activated prompt.

        History is replayed as an undo stack: an ``activate`` row pushes the
        prompt (consecutive duplicates are compressed), a ``rollback`` row
        pops.  Raises ``RuntimeError("no previous activation")`` when there is
        no earlier distinct prompt to return to.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT prompt_id, action FROM activation_history "
                "WHERE program = ? ORDER BY id",
                (program,),
            ).fetchall()
            stack: list[str] = []
            for prompt_id, action in rows:
                if action == "activate":
                    if not stack or stack[-1] != prompt_id:
                        stack.append(prompt_id)
                elif stack:  # rollback
                    stack.pop()
            if len(stack) < 2:
                raise RuntimeError("no previous activation")
            target = stack[-2]
            ts = _now()
            self._conn.execute(
                "INSERT OR REPLACE INTO active "
                "(program, prompt_id, activated_at, gate_report_json) "
                "VALUES (?, ?, ?, ?)",
                (program, target, ts, "{}"),
            )
            self._conn.execute(
                "INSERT INTO activation_history "
                "(program, prompt_id, activated_at, action) VALUES (?, ?, ?, ?)",
                (program, target, ts, "rollback"),
            )
            self._conn.commit()
        return target

    # ------------------------------------------------------------------
    # Lineage
    # ------------------------------------------------------------------

    def lineage(self, prompt_id: str) -> list[str]:
        """Ancestor ids of *prompt_id* via ``parent_ids`` (BFS discovery order)."""
        with self._lock:
            order: list[str] = []
            seen: set[str] = {prompt_id}
            queue: deque[str] = deque([prompt_id])
            while queue:
                current = queue.popleft()
                row = self._conn.execute(
                    "SELECT parent_ids_json FROM prompts WHERE id = ?", (current,)
                ).fetchone()
                if row is None:
                    continue
                for parent in json.loads(row[0]):
                    if parent not in seen:
                        seen.add(parent)
                        order.append(parent)
                        queue.append(parent)
        return order
