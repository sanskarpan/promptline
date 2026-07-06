from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(
    deltas: list[float] | np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval on the mean of paired deltas.

    Parameters
    ----------
    deltas:
        Per-example score differences (candidate − baseline).
    n_boot:
        Number of bootstrap resamples.
    alpha:
        Significance level; produces a ``(1 − alpha)`` CI.
    rng_seed:
        Seed for reproducibility.

    Returns
    -------
    (mean_delta, ci_low, ci_high)
        Mean of the observed deltas plus the lower and upper CI bounds.

    Raises
    ------
    ValueError
        When *deltas* is empty.
    """
    arr = np.asarray(deltas, dtype=float)
    if arr.size == 0:
        raise ValueError("deltas must not be empty")
    rng = np.random.default_rng(rng_seed)
    n = arr.size
    boot_means: np.ndarray = np.mean(rng.choice(arr, size=(n_boot, n), replace=True), axis=1)
    mean_delta = float(np.mean(arr))
    ci_low = float(np.percentile(boot_means, 100.0 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2)))
    return mean_delta, ci_low, ci_high


def bootstrap_pvalue(
    deltas: list[float] | np.ndarray,
    n_boot: int = 10_000,
    rng_seed: int = 0,
) -> float:
    """Two-sided bootstrap p-value for the hypothesis that the mean delta is zero.

    The observed deltas are centred at zero before resampling so the null
    distribution is over a world where the true mean is exactly 0.

    Formula: ``p = (1 + #{|boot_mean| >= |obs_mean|}) / (n_boot + 1)``

    Parameters
    ----------
    deltas:
        Per-example score differences.
    n_boot:
        Number of bootstrap resamples.
    rng_seed:
        Seed for reproducibility.

    Returns
    -------
    float
        Two-sided p-value.

    Raises
    ------
    ValueError
        When *deltas* is empty.
    """
    arr = np.asarray(deltas, dtype=float)
    if arr.size == 0:
        raise ValueError("deltas must not be empty")
    rng = np.random.default_rng(rng_seed)
    n = arr.size
    obs_mean = float(np.mean(arr))
    # Centre deltas at 0 for the null distribution.
    centred = arr - obs_mean
    boot_means: np.ndarray = np.mean(rng.choice(centred, size=(n_boot, n), replace=True), axis=1)
    count = int(np.sum(np.abs(boot_means) >= abs(obs_mean)))
    return (1 + count) / (n_boot + 1)


def holm_correct(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down multiple-testing correction.

    Parameters
    ----------
    pvals:
        Observed p-values in any order.
    alpha:
        Family-wise error rate.

    Returns
    -------
    list[bool]
        Rejection mask in the **original** order of *pvals*.
        ``True`` means the null hypothesis is rejected.
    """
    if not pvals:
        return []
    n = len(pvals)
    # Sort by ascending p-value, keeping original indices.
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    reject = [False] * n
    for step, (orig_idx, pval) in enumerate(indexed):
        threshold = alpha / (n - step)
        if pval <= threshold:
            reject[orig_idx] = True
        else:
            # Step-down: once we fail to reject, stop.
            break
    return reject


def min_examples_warning(n: int, floor: int = 50) -> str | None:
    """Return a warning string when the sample size is small.

    Parameters
    ----------
    n:
        Number of evaluation examples.
    floor:
        Minimum recommended sample size.

    Returns
    -------
    str or None
        A human-readable warning when ``n < floor``, otherwise ``None``.
    """
    if n < floor:
        return f"Warning: only {n} examples (fewer than {floor}); estimates may be unreliable."
    return None
