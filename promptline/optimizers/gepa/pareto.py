"""Pareto-based candidate selection (GEPA Algorithm 2, arXiv:2507.19457).

Pure functions over a score matrix ``scores[candidate_id][i]`` where *i*
indexes instances of ``D_pareto``.  Selection keeps the *per-instance* Pareto
frontier — every candidate that is best on at least one instance survives —
rather than a single global best, preserving complementary "specialist"
candidates as stepping stones.
"""

from __future__ import annotations

import random


def dominates(a: list[float], b: list[float]) -> bool:
    """True iff *a* Pareto-dominates *b*: ``>=`` everywhere and ``>`` somewhere."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def instance_frontiers(scores: dict[str, list[float]]) -> list[set[str]]:
    """Per-instance best sets: ``P*[i] = {c : S[c][i] == max_c' S[c'][i]}``."""
    if not scores:
        return []
    n = len(next(iter(scores.values())))
    frontiers: list[set[str]] = []
    for i in range(n):
        best = max(vec[i] for vec in scores.values())
        frontiers.append({cid for cid, vec in scores.items() if vec[i] == best})
    return frontiers


def frontier_candidates(scores: dict[str, list[float]]) -> set[str]:
    """Union of per-instance best sets, with Pareto-dominated candidates removed."""
    frontiers = instance_frontiers(scores)
    pool: set[str] = set().union(*frontiers) if frontiers else set()
    surviving = {
        c
        for c in pool
        if not any(dominates(scores[other], scores[c]) for other in pool if other != c)
    }
    return surviving


def pareto_sample(scores: dict[str, list[float]], rng: random.Random) -> str:
    """Sample a candidate id from the pruned per-instance Pareto frontier.

    Frequencies are proportional to ``f[c]`` = number of instances on which
    *c* attains the instance-best score, counted over the surviving
    (non-dominated) frontier members.
    """
    if not scores:
        raise ValueError("pareto_sample requires a non-empty score matrix")
    if len(scores) == 1:
        return next(iter(scores))

    surviving = frontier_candidates(scores)
    frontiers = instance_frontiers(scores)

    # Deterministic ordering for reproducible seeded sampling.
    ordered = sorted(surviving)
    weights = [sum(1 for front in frontiers if cid in front) for cid in ordered]

    if not ordered or sum(weights) == 0:
        # Degenerate (e.g. zero-instance matrix): uniform over the pool.
        return rng.choice(sorted(scores))
    return rng.choices(ordered, weights=weights, k=1)[0]
