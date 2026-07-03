"""Mutable run state for the GEPA optimizer.

Holds the candidate pool, the per-instance Pareto score matrix and the loop
counters, plus (de)serialization used for checkpointing.
"""
from __future__ import annotations

from promptline.core.types import Candidate


class GepaState:
    """Candidate pool + per-instance score matrix for a GEPA run.

    Attributes
    ----------
    pool:
        Mapping from candidate id to :class:`Candidate` (insertion-ordered).
    scores:
        ``scores[candidate_id][i]`` is the candidate's score on the *i*-th
        instance of ``D_pareto``.
    """

    def __init__(self) -> None:
        self.pool: dict[str, Candidate] = {}
        self.scores: dict[str, list[float]] = {}
        self.partial: set[str] = set()
        self.iteration: int = 0
        self.merges_done: int = 0
        self.accepted_count: int = 0
        self.accepts_since_merge: int = 0
        self.module_counter: int = 0

    # ------------------------------------------------------------------
    # Pool operations
    # ------------------------------------------------------------------

    def add(self, candidate: Candidate, pareto_scores: list[float]) -> None:
        """Add *candidate* with its per-instance ``D_pareto`` scores."""
        self.pool[candidate.id] = candidate
        self.scores[candidate.id] = list(pareto_scores)

    def mean(self, candidate_id: str) -> float:
        """Mean score of a candidate over ``D_pareto`` (0.0 when empty)."""
        vec = self.scores.get(candidate_id, [])
        if not vec:
            return 0.0
        return sum(vec) / len(vec)

    def best_id(self) -> str:
        """Id of the candidate with the highest mean ``D_pareto`` score."""
        return max(self.pool, key=self.mean)

    # ------------------------------------------------------------------
    # Checkpoint (de)serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict for checkpointing."""
        return {
            "candidates": [c.model_dump() for c in self.pool.values()],
            "scores": {cid: list(vec) for cid, vec in self.scores.items()},
            "partial": list(self.partial),
            "iteration": self.iteration,
            "merges_done": self.merges_done,
            "accepted_count": self.accepted_count,
            "accepts_since_merge": self.accepts_since_merge,
            "module_counter": self.module_counter,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GepaState:
        """Restore a :class:`GepaState` from :meth:`to_dict` output."""
        state = cls()
        for raw in data.get("candidates", []):
            candidate = Candidate.model_validate(raw)
            state.pool[candidate.id] = candidate
        state.scores = {
            cid: [float(s) for s in vec]
            for cid, vec in data.get("scores", {}).items()
        }
        state.partial = set(data.get("partial", []))
        state.iteration = int(data.get("iteration", 0))
        state.merges_done = int(data.get("merges_done", 0))
        state.accepted_count = int(data.get("accepted_count", 0))
        state.accepts_since_merge = int(data.get("accepts_since_merge", 0))
        state.module_counter = int(data.get("module_counter", 0))
        return state
