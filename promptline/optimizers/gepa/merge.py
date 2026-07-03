"""System-aware merge (crossover) for GEPA — Appendix F of arXiv:2507.19457.

Two frontier candidates from different lineages are recombined module-by-module
against their common ancestor using the "triplet rule": a parent that diverged
from the ancestor is assumed to carry a useful learned mutation for that
module.
"""
from __future__ import annotations

import random
from collections import deque

from promptline.core.types import Candidate, ModuleState


def ancestor_ids(candidate_id: str, pool: dict[str, Candidate]) -> set[str]:
    """All (transitive) proper ancestors of *candidate_id* found in *pool*."""
    seen: set[str] = set()
    queue: deque[str] = deque(
        pool[candidate_id].parent_ids if candidate_id in pool else []
    )
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        if current in pool:
            queue.extend(pool[current].parent_ids)
    return seen


def is_related(id1: str, id2: str, pool: dict[str, Candidate]) -> bool:
    """True when one candidate is an ancestor (or self) of the other."""
    if id1 == id2:
        return True
    return id1 in ancestor_ids(id2, pool) or id2 in ancestor_ids(id1, pool)


def common_ancestor(
    id1: str,
    id2: str,
    pool: dict[str, Candidate],
) -> str | None:
    """Nearest common ancestor of two candidates via BFS on ``parent_ids``.

    Returns ``None`` when the lineages never meet (e.g. independent seeds) or
    when the ancestor is not present in *pool*.
    """
    side1 = ancestor_ids(id1, pool) | {id1}
    # BFS from id2 upward; the first hit in side1 is the nearest common ancestor.
    seen: set[str] = set()
    queue: deque[str] = deque([id2])
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        if current in side1 and current in pool:
            return current
        if current in pool:
            queue.extend(pool[current].parent_ids)
    return None


def merge_candidates(
    parent1: Candidate,
    parent2: Candidate,
    ancestor: Candidate,
    mean1: float,
    mean2: float,
    rng: random.Random,
    optimizer: str = "gepa",
) -> Candidate:
    """Recombine two parents against their common *ancestor* (triplet rule).

    Per module: exactly one parent differs from the ancestor → take that
    parent's :class:`ModuleState`; both differ → take the higher-mean parent's
    (seeded rng tie-break); neither differs → parent1's.
    """
    merged: dict[str, ModuleState] = {}
    for name in parent1.modules:
        m1 = parent1.modules[name]
        m2 = parent2.modules[name]
        ma = ancestor.modules[name]
        d1 = m1 != ma
        d2 = m2 != ma
        if d1 and not d2:
            pick = m1
        elif d2 and not d1:
            pick = m2
        elif d1 and d2:
            if mean1 > mean2:
                pick = m1
            elif mean2 > mean1:
                pick = m2
            else:
                pick = rng.choice([m1, m2])
        else:
            pick = m1
        merged[name] = pick.model_copy(deep=True)
    return parent1.child(
        modules=merged, optimizer=optimizer, extra_parents=(parent2.id,)
    )
