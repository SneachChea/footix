from typing import List, Tuple, Union

import numpy as np


def entropy(proba: list[float] | np.ndarray, outcome_idx: int) -> float:
    """
    Compute the entropy (or incertity) metric.

    Args:
        proba (Union[List, np.ndarray]): list of probabilities
        outcome_idx (int): index of the outcome, can be 0, 1, 2 for Home, Draw and Away

    Returns:
        float: entropy metrics
    """
    p_r = proba[outcome_idx]
    return -np.log(p_r) / np.log(3)


def rps(probas: list[float] | np.ndarray, outcome_idx: int) -> float:
    """
        Compute the Ranked Probability Score.

    Args:
        probas (Union[List, np.ndarray]): list of probabilities
        outcome_idx (int): index of the outcome. 0, 1, 2 for Home, Draw and Away

    Returns:
        float: RPS metrics
    """
    outcome = np.zeros_like(probas)
    outcome[outcome_idx] = 1.0
    cum_probas = np.cumsum(probas)
    cum_outcome = np.cumsum(outcome)
    sum_rps = 0
    for i in range(len(outcome)):
        sum_rps += (cum_probas[i] - cum_outcome[i]) ** 2

    return sum_rps / (len(outcome) - 1)


def zscore(probas: list[float] | np.ndarray, rps_real: float) -> Tuple[float, float, float]:
    """
        Compute the Z-score in respect of the RPS computed

    Args:
        probas (Union[List, np.ndarray]): list of probabilities
        RPS_real (float): RPS result

    Returns:
        float: Z-score
    """

    eps = 1e-5
    n_iter = 100

    def _monteCarl(probas: Union[List, np.ndarray], n_iter: int) -> tuple[float, float]:
        outcomes = [0, 1, 2]
        rps_stats = np.zeros(n_iter)

        if np.sum(probas) != 1.0:
            probas = probas / np.sum(probas)
        for i in range(n_iter):
            res = np.random.choice(outcomes, p=probas)
            rps_stats[i] = rps(probas, res)
        return np.mean(rps_stats), np.std(rps_stats)  # type: ignore

    mu, sigma = _monteCarl(probas, n_iter)

    return (rps_real - mu) / (sigma + eps), mu, sigma
