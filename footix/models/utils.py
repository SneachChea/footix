from typing import Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from scipy.optimize import least_squares, root
from scipy.special import iv

import footix.utils.decorators as decorators
from footix.models.score_matrix import GoalMatrix
from footix.utils.typing import ProbaResult


@decorators.verify_required_column(column_names=["home_team", "fthg"])
def compute_goals_home_vectors(
    data: pd.DataFrame, /, map_teams: dict, nbr_team: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute vectors representing home team goals.

    Args:
        data (pd.DataFrame): Input DataFrame with home team goals and HomeTeam column.
        map_teams (dict): Dictionary mapping team names to numerical IDs.
        nbr_team (int): Number of teams in the league.
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            x representing home team goals and tau_home representing binary vectors
            for each home team.

    """
    x = np.zeros(len(data))
    tau_home = np.zeros((len(data), nbr_team))
    for i, row in data.iterrows():
        j = map_teams[row["home_team"]]
        x[i] = row["fthg"]
        tau_home[i, j] = 1
    return x, tau_home


@decorators.verify_required_column(column_names=["away_team", "ftag"])
def compute_goals_away_vectors(
    data: pd.DataFrame, /, map_teams: dict[str, int], nbr_team: int
) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros(len(data))
    tau_away = np.zeros((len(data), nbr_team))
    for i, row in data.iterrows():
        j = map_teams[row["away_team"]]
        x[i] = row["ftag"]
        tau_away[i, j] = 1
    return x, tau_away


def to_torch_tensor(
    *arrays: np.ndarray, dtype: torch.dtype = torch.float32
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Convert numpy arrays to torch tensors.

    Args:
        *arrays: Variable number of numpy arrays to convert
        dtype: Target tensor dtype (default: torch.float32)
    Returns:
        Single tensor if one array is provided, tuple of tensors if multiple arrays
    Examples:
        >>> x = np.array([1, 2, 3])
        >>> tensor_x = to_tensor(x)

        >>> x = np.array([1, 2, 3])
        >>> y = np.array([4, 5, 6])
        >>> tensor_x, tensor_y = to_tensor(x, y)

    """
    tensors = tuple(torch.from_numpy(arr).type(dtype) for arr in arrays)
    return tensors[0] if len(tensors) == 1 else tensors


def poisson_proba(lambda_param: float, k: int) -> np.ndarray:
    """Calculate the probability of achieving up to k goals given a lambda parameter.

    Args:
        lambda_param (float): The expected number of goals.
        k (int): The number of goals to achieve.

    Returns:
        np.ndarray:  An array containing the probabilities of achieving each possible
    number of goals from 0 to n_goals, inclusive.

    """
    poisson = stats.poisson(mu=lambda_param)
    k_list = np.arange(k)
    return poisson.pmf(k=k_list)  # type:ignore


def implicit_intensities(
    proba_from_odds: np.ndarray, max_iter: int = 200, tol: float = 1e-10
) -> np.ndarray:
    """Calculate implicit scoring intensities from match outcome probabilities.

    This function converts betting odds probabilities into implied goal-scoring
    intensities (lambda parameters) for both teams using numerical optimization.
    It uses the Skellam distribution to model the difference between two Poisson
    processes (goal scoring by each team).

    Args:
        proba_from_odds (np.ndarray): Array of shape (n_matches, 3) containing
            probabilities for [win, draw, loss] derived from betting odds.
        max_iter (int, optional): Maximum number of iterations for the optimization
            algorithm. Defaults to 200.
        tol (float, optional): Tolerance for optimization convergence. Defaults to 1e-10.

    Raises:
        ValueError: If proba_from_odds does not have shape (n_matches, 3).

    Returns:
        np.ndarray: Array of shape (n_matches, 2) containing the implied scoring
            intensities [lambda1, lambda2] for each match, where lambda1 is the
            home team's scoring intensity and lambda2 is the away team's.

    Note:
        If the primary optimization fails, the function falls back to a grid search
        over predefined lambda values to find the best approximation.

    """
    proba_from_odds = np.asarray(proba_from_odds, dtype=float)
    eps = 1e-12
    if proba_from_odds.ndim != 2 or proba_from_odds.shape[1] != 3:
        raise ValueError("`pi` doit avoir la forme (n_matches, 3).")

    p = np.clip(proba_from_odds, eps, 1 - eps)
    row_sums = p.sum(axis=1, keepdims=True)
    p /= row_sums

    results = np.empty((p.shape[0], 2), dtype=float)
    lg = np.logspace(-2, 2, 50)  # maillage pour le fallback

    for i, (p_w, p_d, p_l) in enumerate(p):
        target = np.array([p_w + p_d, p_l])

        mu_diff = p_w - p_l
        lam0 = max(0.2, 1.0 + mu_diff)
        lam1 = max(0.2, 1.0 - mu_diff)
        x0 = np.array([lam0, lam1])

        def residual(t):
            lam1, lam2 = t
            p_wd = 1 - stats.skellam.cdf(-1, lam1, lam2)  # P(Y1 ≥ Y2)
            p_l = stats.skellam.cdf(-1, lam1, lam2)  # P(Y1 <  Y2)
            return (np.array([p_wd, p_l]) - target) / np.sqrt(target * (1 - target))

        sol = least_squares(
            residual, x0, bounds=(1e-6, np.inf), xtol=tol, ftol=tol, gtol=tol, max_nfev=max_iter
        )

        if sol.success and np.all(sol.x > 0):
            results[i] = sol.x
            continue
        best_err, best_t = np.inf, x0
        for t1 in lg:
            for t2 in lg:
                err = np.sum(residual([t1, t2]) ** 2)
                if err < best_err:
                    best_err, best_t = err, (t1, t2)  # type: ignore
        results[i] = best_t

    return results


def _p0(lamda_1: float, lamda_2: float) -> float:
    return np.exp(-(lamda_1 + lamda_2)) * iv(0, 2 * np.sqrt(lamda_1 * lamda_2))


def _p_pos(lamda_1: float, lamda_2: float, K: int = 40) -> float:
    k = np.arange(1, K + 1)
    return np.sum(
        np.exp(-(lamda_1 + lamda_2))
        * ((lamda_1 / lamda_2) ** (k / 2) * iv(k, 2 * np.sqrt(lamda_1 * lamda_2)))
    )


def implied_poisson_goals(
    bookmaker_proba: ProbaResult, *, k_sum: int = 40, nbr_goals: int = 10
) -> GoalMatrix:
    """Calculate implied Poisson goal distributions from bookmaker probabilities.

    This function uses a system of equations to find the Poisson parameters (lambda)
    that best match the observed probabilities from bookmakers. It solves for the
    scoring rates of both teams using modified Bessel functions of the first kind.

    Args:
        bookmaker_proba: Probabilities from bookmaker (draw, home win, away win)
        k_sum: Maximum number of goals to consider in summation (default: 40)
        nbr_goals: Number of goals to generate probabilities for (default: 10)

    Returns:
        GoalMatrix containing probability distributions for home and away goals

    Raises:
        ArithmeticError: If the numerical solver fails to converge

    """
    proba_draw = bookmaker_proba.proba_draw
    proba_home_win = bookmaker_proba.proba_home

    def system(params: np.ndarray, p_0_obs: float, p_pos_obs: float) -> list[float]:
        """System of equations to solve for Poisson parameters.

        Args:
            params: Log of lambda parameters [log(λ1), log(λ2)]
            p_0_obs: Observed probability of draw
            p_pos_obs: Observed probability of home win

        Returns:
            Differences between model and observed probabilities

        """
        l1, l2 = np.exp(params)
        p_0_model = _p0(l1, l2)  # Probability of draw
        p_pos_model = _p_pos(l1, l2, K=k_sum)  # Probability of home win
        return [p_0_model - p_0_obs, p_pos_model - p_pos_obs]

    # Initial guess for lambda parameters (log scale)
    initial_guess = [np.log(1.2), np.log(0.9)]

    # Solve system of equations
    sol = root(system, x0=initial_guess, args=(proba_draw, proba_home_win))

    if not sol.success:
        raise ArithmeticError("Numerical solver failed to converge")

    # Convert solution back from log scale
    lamda_1, lamda_2 = np.exp(sol.x)

    return GoalMatrix(
        home_goals_probs=poisson_proba(lambda_param=lamda_1, k=nbr_goals),
        away_goals_probs=poisson_proba(lambda_param=lamda_2, k=nbr_goals),
    )
