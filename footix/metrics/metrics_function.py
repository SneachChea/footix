import numpy as np
from footix.utils.typing import ArrayLikeF

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
    """Compute the Ranked Probability Score.

    Args:
        probas ArrayLike: list of probabilities
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


def zscore(
    probas: ArrayLikeF, rps_real: float, seed: int | None = None, n_iter: int = 1000
) -> tuple[float, float, float]:
    """Compute the Z-score in respect of the RPS computed. The z-score shows how many standard
    deviations the observed RPS was away from what could have been expected, if the probabilities
    of each model were perfect.

    Args:
        probas ArrayLike: list of probabilities
        RPS_real (float): RPS result
        seed (int or None): seed for Monte-Carlo computation
        n_iter (int): number of iteration for Monte-Carlo simulation
    Returns:
        float: Z-score
        float: mu
        float: sigma

    """

    _eps = 1e-5

    def _monteCarl(
        probas: ArrayLikeF, n_iter: int, seed: int | None
    ) -> tuple[float, float]:
        outcomes = [0, 1, 2]
        rps_stats = np.zeros(n_iter)
        rng = np.random.default_rng(seed=seed)
        if np.sum(probas) != 1.0:
            probas = probas / np.sum(probas)
        for i in range(n_iter):
            res = rng.choice(outcomes, p=probas)
            rps_stats[i] = rps(probas, res)
        return np.mean(rps_stats), np.std(rps_stats)  # type: ignore

    mu, sigma = _monteCarl(probas, n_iter=n_iter, seed=seed)

    return (rps_real - mu) / (sigma + _eps), mu, sigma
