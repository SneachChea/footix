from typing import NamedTuple, Optional

import numpy as np

from footix.utils.typing import ArrayLikeF


class RPSResult(NamedTuple):
    """Named tuple for Ranked Probability Score statistics."""

    z_score: float
    mean: float
    std_dev: float


def incertity(probas: ArrayLikeF, outcome_idx: int) -> float:
    """Compute the entropy (or incertity) metric.

    Args:
        proba ArrayLike float: list of probabilities
        outcome_idx (int): index of the outcome, can be 0, 1, 2 for Home, Draw and Away

    Returns:
        float: entropy metrics

    """
    p_r = probas[outcome_idx]
    return -np.log(p_r) / np.log(3)


def rps(probas: ArrayLikeF, outcome_idx: int) -> float:
    """Compute the Ranked Probability Score (RPS) for a single categorical forecast.

    RPS measures the squared differences between cumulative forecast probabilities and
    the cumulative actual outcome. Lower scores indicate better forecasts.

    Args:
        probas: Sequence of forecast probabilities for each category (must sum to 1).
        outcome_idx: Index of the realized outcome (0-based).

    Returns:
        The RPS value.

    Raises:
        ValueError: If probabilities are invalid or outcome_idx is out of range.

    """
    probas_arr = np.asarray(probas, dtype=float)
    n_categories = probas_arr.size

    if not np.all(probas_arr >= 0):
        raise ValueError("Probabilities must be non-negative.")
    total = probas_arr.sum()
    if not np.isclose(total, 1.0):
        probas_arr = probas_arr / total

    if not (0 <= outcome_idx < n_categories):
        raise ValueError(f"outcome_idx must be between 0 and {n_categories - 1}.")

    # One-hot encode the observed outcome
    outcome = np.zeros(n_categories, dtype=float)
    outcome[outcome_idx] = 1.0

    # Compute cumulative sums and RPS
    cum_probas = np.cumsum(probas_arr)
    cum_outcome = np.cumsum(outcome)
    squared_diffs = (cum_probas - cum_outcome) ** 2

    # Average over the first (n-1) categories
    return np.sum(squared_diffs) / (n_categories - 1)


def zscore(
    probas: ArrayLikeF, rps_observed: float, n_iter: int = 10_000, seed: Optional[int] = None
) -> RPSResult:
    """Compute the z-score of an observed RPS against a Monte Carlo distribution.

    This quantifies how many standard deviations the observed RPS is from the
    expected RPS if forecasts were perfect probabilistically.

    Args:
        probas: Sequence of forecast probabilities for each category (must sum to 1).
        rps_observed: The observed RPS value to evaluate.
        n_iter: Number of Monte Carlo samples (default: 10000).
        seed: Random seed for reproducibility.

    Returns:
        RPSResult: A tuple containing (z_score, mean_rps, std_rps).

    """
    rng = np.random.default_rng(seed)
    probas_arr = np.asarray(probas, dtype=float)

    if not np.all(probas_arr >= 0):
        raise ValueError("Probabilities must be non-negative.")
    total = probas_arr.sum()
    if not np.isclose(total, 1.0, rtol=1e-12):
        probas_arr = probas_arr / probas_arr.sum()
        probas_arr = np.round(probas_arr, decimals=10)
    categories = np.arange(probas_arr.size)
    samples = rng.choice(categories, size=n_iter, p=probas_arr)

    # Vectorized RPS computation for all samples
    # One-hot matrix: shape (n_iter, n_categories)
    one_hot = np.zeros((n_iter, probas_arr.size), dtype=float)
    one_hot[np.arange(n_iter), samples] = 1.0

    cum_probas = np.cumsum(probas_arr)
    cum_outcome = np.cumsum(one_hot, axis=1)
    diffs = cum_probas - cum_outcome
    rps_vals = np.sum(diffs**2, axis=1) / (probas_arr.size - 1)

    mean_rps = rps_vals.mean()
    std_rps = rps_vals.std(ddof=1)
    # Avoid division by zero
    if std_rps == 0:
        z_score = np.inf if rps_observed > mean_rps else -np.inf
    else:
        z_score = (rps_observed - mean_rps) / std_rps

    return RPSResult(z_score=z_score, mean=mean_rps, std_dev=std_rps)
