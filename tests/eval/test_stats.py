from __future__ import annotations

import numpy as np
import pytest

from promptline.eval.stats import (
    bootstrap_pvalue,
    holm_correct,
    min_examples_warning,
    paired_bootstrap_ci,
)

# ---------------------------------------------------------------------------
# paired_bootstrap_ci
# ---------------------------------------------------------------------------


def test_ci_excludes_zero_for_clear_positive_effect() -> None:
    """N(0.1, 0.1, n=500): the 95% CI should not contain 0."""
    rng = np.random.default_rng(0)
    deltas = rng.normal(loc=0.1, scale=0.1, size=500).tolist()
    mean_d, ci_low, ci_high = paired_bootstrap_ci(deltas, rng_seed=1)
    assert ci_low > 0.0, f"Expected CI to exclude 0 but got [{ci_low}, {ci_high}]"


def test_ci_includes_zero_for_null_effect() -> None:
    """N(0, 1, n=30): the 95% CI should span 0 with high probability."""
    rng = np.random.default_rng(42)
    deltas = rng.normal(loc=0.0, scale=1.0, size=30).tolist()
    mean_d, ci_low, ci_high = paired_bootstrap_ci(deltas, rng_seed=2)
    assert ci_low <= 0.0 <= ci_high, f"Expected CI to include 0 but got [{ci_low}, {ci_high}]"


def test_ci_all_zeros_returns_zero_triple() -> None:
    deltas = [0.0] * 20
    mean_d, ci_low, ci_high = paired_bootstrap_ci(deltas)
    assert mean_d == pytest.approx(0.0)
    assert ci_low == pytest.approx(0.0)
    assert ci_high == pytest.approx(0.0)


def test_ci_empty_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap_ci([])


def test_ci_mean_matches_sample_mean() -> None:
    deltas = [0.1, 0.2, 0.3, 0.4]
    mean_d, _, _ = paired_bootstrap_ci(deltas)
    assert mean_d == pytest.approx(0.25)


def test_ci_low_le_mean_le_high() -> None:
    rng = np.random.default_rng(7)
    deltas = rng.normal(0.5, 0.2, 100).tolist()
    mean_d, ci_low, ci_high = paired_bootstrap_ci(deltas)
    assert ci_low <= mean_d <= ci_high


# ---------------------------------------------------------------------------
# bootstrap_pvalue
# ---------------------------------------------------------------------------


def test_pvalue_null_deltas_mostly_above_threshold() -> None:
    """For truly null deltas, p > 0.05 for at least 17 of 20 seeds."""
    outer_rng = np.random.default_rng(99)
    above = 0
    for seed in range(20):
        deltas = outer_rng.normal(0.0, 1.0, 100).tolist()
        p = bootstrap_pvalue(deltas, rng_seed=seed)
        if p > 0.05:
            above += 1
    assert above >= 17, f"Only {above}/20 seeds gave p > 0.05 for null deltas"


def test_pvalue_strong_effect_below_threshold() -> None:
    """N(1, 0.1, n=100): p-value should be very small."""
    rng = np.random.default_rng(0)
    deltas = rng.normal(1.0, 0.1, 100).tolist()
    p = bootstrap_pvalue(deltas, rng_seed=0)
    assert p < 0.01, f"Expected p < 0.01 for strong effect but got p={p}"


def test_pvalue_all_zeros_returns_one() -> None:
    """All-zero deltas: every bootstrap mean equals |obs_mean| = 0 → p = 1.0."""
    deltas = [0.0] * 50
    p = bootstrap_pvalue(deltas)
    assert p == pytest.approx(1.0)


def test_pvalue_empty_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_pvalue([])


def test_pvalue_in_range() -> None:
    """p-value must lie in (0, 1]."""
    rng = np.random.default_rng(5)
    deltas = rng.normal(0.2, 0.5, 50).tolist()
    p = bootstrap_pvalue(deltas)
    assert 0.0 < p <= 1.0


# ---------------------------------------------------------------------------
# holm_correct
# ---------------------------------------------------------------------------


def test_holm_spec_example() -> None:
    """[0.01, 0.04, 0.03] with alpha=0.05 → [True, False, False]."""
    result = holm_correct([0.01, 0.04, 0.03], alpha=0.05)
    assert result == [True, False, False]


def test_holm_empty_returns_empty() -> None:
    assert holm_correct([]) == []


def test_holm_all_rejected_when_all_small() -> None:
    result = holm_correct([0.001, 0.001, 0.001], alpha=0.05)
    assert all(result)


def test_holm_none_rejected_when_all_large() -> None:
    result = holm_correct([0.5, 0.6, 0.7], alpha=0.05)
    assert not any(result)


def test_holm_preserves_original_order() -> None:
    """The rejection mask must correspond to the input order, not sorted order."""
    # p-values in reverse sorted order; only smallest (index 2) should be rejected
    pvals = [0.8, 0.5, 0.001]
    result = holm_correct(pvals, alpha=0.05)
    assert result[2] is True
    assert result[0] is False
    assert result[1] is False


def test_holm_single_pvalue_rejected() -> None:
    assert holm_correct([0.01], alpha=0.05) == [True]


def test_holm_single_pvalue_not_rejected() -> None:
    assert holm_correct([0.10], alpha=0.05) == [False]


# ---------------------------------------------------------------------------
# min_examples_warning
# ---------------------------------------------------------------------------


def test_min_examples_warning_below_floor() -> None:
    msg = min_examples_warning(30)
    assert msg is not None
    assert "30" in msg


def test_min_examples_warning_at_floor_returns_none() -> None:
    assert min_examples_warning(50) is None


def test_min_examples_warning_above_floor_returns_none() -> None:
    assert min_examples_warning(100) is None


def test_min_examples_warning_custom_floor() -> None:
    assert min_examples_warning(90, floor=100) is not None
    assert min_examples_warning(100, floor=100) is None
