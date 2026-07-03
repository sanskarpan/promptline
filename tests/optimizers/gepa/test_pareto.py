"""Tests for GEPA Pareto selection (Algorithm 2) and GepaState."""
from __future__ import annotations

import random
from collections import Counter

from promptline.core.types import Candidate, ModuleState
from promptline.optimizers.gepa.pareto import (
    dominates,
    frontier_candidates,
    instance_frontiers,
    pareto_sample,
)
from promptline.optimizers.gepa.state import GepaState

# ---------------------------------------------------------------------------
# dominates
# ---------------------------------------------------------------------------


def test_dominates_strict() -> None:
    assert dominates([1.0, 1.0], [1.0, 0.5])
    assert not dominates([1.0, 0.5], [1.0, 1.0])
    # Equal vectors do not dominate each other.
    assert not dominates([1.0, 1.0], [1.0, 1.0])
    # Incomparable vectors: neither dominates.
    assert not dominates([1.0, 0.0], [0.0, 1.0])
    assert not dominates([0.0, 1.0], [1.0, 0.0])


# ---------------------------------------------------------------------------
# pareto_sample
# ---------------------------------------------------------------------------


def _draws(scores: dict[str, list[float]], n: int = 200) -> Counter:
    counts: Counter = Counter()
    for seed in range(n):
        counts[pareto_sample(scores, random.Random(seed))] += 1
    return counts


def test_single_dominant_candidate_always_sampled() -> None:
    scores = {
        "dom": [1.0, 1.0, 1.0],
        "a": [0.5, 0.9, 0.9],
        "b": [0.9, 0.5, 0.9],
    }
    counts = _draws(scores)
    assert counts == Counter({"dom": 200})


def test_specialist_sampled_with_nonzero_frequency() -> None:
    # "gen" is best on instances 0-1; "spec" is uniquely best on instance 2.
    scores = {
        "gen": [1.0, 1.0, 0.0],
        "spec": [0.0, 0.0, 1.0],
        "mid": [0.5, 0.5, 0.5],
    }
    counts = _draws(scores)
    assert counts["spec"] > 0
    assert counts["gen"] > 0
    # "mid" is not instance-best anywhere.
    assert counts["mid"] == 0


def test_dominated_candidate_never_sampled() -> None:
    scores = {
        "a": [1.0, 0.0],
        "b": [0.0, 1.0],
        "dominated": [0.0, 0.5],
    }
    counts = _draws(scores)
    assert counts["dominated"] == 0
    assert counts["a"] > 0 and counts["b"] > 0


def test_tied_candidates_both_sampleable() -> None:
    scores = {
        "x": [1.0, 1.0],
        "y": [1.0, 1.0],
    }
    counts = _draws(scores)
    assert counts["x"] > 0 and counts["y"] > 0


def test_single_candidate_pool() -> None:
    assert pareto_sample({"only": [0.0, 0.0]}, random.Random(0)) == "only"


# ---------------------------------------------------------------------------
# frontier helpers
# ---------------------------------------------------------------------------


def test_instance_frontiers_ties() -> None:
    scores = {"a": [1.0, 0.0], "b": [1.0, 1.0]}
    fronts = instance_frontiers(scores)
    assert fronts == [{"a", "b"}, {"b"}]


def test_frontier_candidates_prunes_dominated_frontier_member() -> None:
    # "a" ties the max on instance 0 but is dominated by "b" overall.
    scores = {"a": [1.0, 0.0], "b": [1.0, 1.0]}
    assert frontier_candidates(scores) == {"b"}


# ---------------------------------------------------------------------------
# GepaState (de)serialization
# ---------------------------------------------------------------------------


def _candidate(instruction: str) -> Candidate:
    return Candidate.seed(modules={"main": ModuleState(instruction=instruction)})


def test_state_roundtrip() -> None:
    state = GepaState()
    seed = _candidate("seed")
    child = seed.child(
        modules={"main": ModuleState(instruction="child")}, optimizer="gepa"
    )
    state.add(seed, [0.5, 0.5])
    state.add(child, [1.0, 0.0])
    state.iteration = 7
    state.merges_done = 1
    state.accepted_count = 2
    state.accepts_since_merge = 3
    state.module_counter = 4

    restored = GepaState.from_dict(state.to_dict())
    assert list(restored.pool) == [seed.id, child.id]
    assert restored.pool[child.id].parent_ids == [seed.id]
    assert restored.scores == state.scores
    assert restored.iteration == 7
    assert restored.merges_done == 1
    assert restored.accepted_count == 2
    assert restored.accepts_since_merge == 3
    assert restored.module_counter == 4


def test_state_mean_and_best() -> None:
    state = GepaState()
    a, b = _candidate("a"), _candidate("b")
    state.add(a, [0.0, 1.0])
    state.add(b, [1.0, 1.0])
    assert state.mean(a.id) == 0.5
    assert state.mean(b.id) == 1.0
    assert state.best_id() == b.id
