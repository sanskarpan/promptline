"""Judge/human agreement metrics.

Implemented directly on numpy (no sklearn dependency).
"""
from __future__ import annotations

from typing import cast

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------


def cohens_kappa(a: list[int], b: list[int], weights: str | None = None) -> float:
    """Cohen's kappa between two integer ratings of the same items.

    Parameters
    ----------
    a, b:
        Equal-length lists of category labels.
    weights:
        ``None`` for classic (unweighted) kappa, or ``"quadratic"`` for
        quadratic-weighted kappa (weights based on label-value distance).
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("cannot compute kappa on empty ratings")
    if weights not in (None, "quadratic"):
        raise ValueError(f"unsupported weights: {weights!r}")

    labels = sorted(set(a) | set(b))
    index = {label: i for i, label in enumerate(labels)}
    k = len(labels)

    observed = np.zeros((k, k), dtype=float)
    for x, y in zip(a, b, strict=True):
        observed[index[x], index[y]] += 1.0
    n = observed.sum()

    if weights is None:
        weight = 1.0 - np.eye(k)
    else:  # quadratic
        values = np.asarray(labels, dtype=float)
        span = values.max() - values.min()
        diff = values[:, None] - values[None, :]
        weight = (diff**2) / (span**2) if span > 0 else np.zeros((k, k))

    row = observed.sum(axis=1)
    col = observed.sum(axis=0)
    expected = np.outer(row, col) / n

    expected_disagreement = float((weight * expected).sum())
    if expected_disagreement == 0.0:
        # Degenerate: a single category (or zero-span labels). Perfect
        # agreement by construction.
        return 1.0
    observed_disagreement = float((weight * observed).sum())
    return 1.0 - observed_disagreement / expected_disagreement


# ---------------------------------------------------------------------------
# Spearman rank correlation
# ---------------------------------------------------------------------------


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation via :func:`scipy.stats.spearmanr`."""
    # scipy types spearmanr's SignificanceResult elements as plain `object`,
    # so cast the statistic; at runtime it is always a numpy float.
    statistic = cast("float", stats.spearmanr(a, b)[0])
    return float(statistic)


# ---------------------------------------------------------------------------
# Pairwise verdict accuracy
# ---------------------------------------------------------------------------


def pairwise_accuracy(
    judge: list[str],
    human: list[str],
    ignore_ties: bool = True,
) -> float:
    """Fraction of A/B/TIE verdicts where the judge matches the human.

    When *ignore_ties* is true, pairs where the human verdict is ``"TIE"``
    are dropped before computing accuracy.  Returns 0.0 when no pairs remain.
    """
    if len(judge) != len(human):
        raise ValueError(f"length mismatch: {len(judge)} vs {len(human)}")
    pairs = list(zip(judge, human, strict=True))
    if ignore_ties:
        pairs = [(j, h) for j, h in pairs if h != "TIE"]
    if not pairs:
        return 0.0
    return sum(1 for j, h in pairs if j == h) / len(pairs)
