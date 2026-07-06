"""Tests for promptline.judge.metrics (Task 16)."""

from __future__ import annotations

import pytest
from scipy import stats

from promptline.judge.metrics import cohens_kappa, pairwise_accuracy, spearman

# ---------------------------------------------------------------------------
# cohens_kappa — unweighted
# ---------------------------------------------------------------------------


def test_kappa_hand_computed_2x2() -> None:
    # confusion (labels 0,1): [[2,0],[1,1]] -> po=0.75, pe=0.5 -> kappa=0.5
    a = [1, 1, 0, 0]
    b = [1, 0, 0, 0]
    assert cohens_kappa(a, b) == pytest.approx(0.5)


def test_kappa_perfect_agreement() -> None:
    a = [0, 1, 2, 1, 0]
    assert cohens_kappa(a, list(a)) == pytest.approx(1.0)


def test_kappa_symmetric() -> None:
    a = [1, 1, 0, 0]
    b = [1, 0, 0, 0]
    assert cohens_kappa(a, b) == pytest.approx(cohens_kappa(b, a))


def test_kappa_single_category_degenerate() -> None:
    assert cohens_kappa([1, 1, 1], [1, 1, 1]) == pytest.approx(1.0)


def test_kappa_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cohens_kappa([1, 2], [1])


# ---------------------------------------------------------------------------
# cohens_kappa — quadratic weights
# ---------------------------------------------------------------------------


def test_quadratic_kappa_hand_computed_3_class() -> None:
    # a=[0,1,2,1], b=[0,2,2,1]
    # w[i,j] = (i-j)^2 / (2-0)^2; sum(wO)=0.25; sum(wE)=1.25 -> kappa=0.8
    a = [0, 1, 2, 1]
    b = [0, 2, 2, 1]
    assert cohens_kappa(a, b, weights="quadratic") == pytest.approx(0.8)


def test_quadratic_kappa_perfect_agreement() -> None:
    a = [1, 2, 3, 4, 5]
    assert cohens_kappa(a, list(a), weights="quadratic") == pytest.approx(1.0)


def test_kappa_unknown_weights_raises() -> None:
    with pytest.raises(ValueError):
        cohens_kappa([0, 1], [0, 1], weights="linear")


# ---------------------------------------------------------------------------
# spearman
# ---------------------------------------------------------------------------


def test_spearman_matches_scipy() -> None:
    a = [1, 2, 3, 4, 5]
    b = [2, 1, 4, 3, 5]
    expected = float(stats.spearmanr(a, b).statistic)
    assert spearman(a, b) == pytest.approx(expected)


def test_spearman_perfect_monotone() -> None:
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)


def test_spearman_perfect_inverse() -> None:
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# pairwise_accuracy
# ---------------------------------------------------------------------------


def test_pairwise_accuracy_ignores_human_ties_by_default() -> None:
    judge = ["A", "B", "A", "TIE"]
    human = ["A", "TIE", "B", "TIE"]
    # kept pairs: (A,A) correct, (A,B) wrong -> 0.5
    assert pairwise_accuracy(judge, human) == pytest.approx(0.5)


def test_pairwise_accuracy_with_ties_counted() -> None:
    judge = ["A", "B", "A", "TIE"]
    human = ["A", "TIE", "B", "TIE"]
    # (A,A) ok, (B,TIE) wrong, (A,B) wrong, (TIE,TIE) ok -> 0.5
    assert pairwise_accuracy(judge, human, ignore_ties=False) == pytest.approx(0.5)


def test_pairwise_accuracy_perfect() -> None:
    assert pairwise_accuracy(["A", "B"], ["A", "B"]) == pytest.approx(1.0)


def test_pairwise_accuracy_all_human_ties_returns_zero() -> None:
    assert pairwise_accuracy(["A", "B"], ["TIE", "TIE"]) == 0.0
