import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import root
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
            goals_vector representing home team goals and tau_home representing binary vectors
            for each home team.

    """
    goals_vector = data["fthg"].values
    tau_home = np.zeros((len(data), nbr_team))
    tau_home[np.arange(len(data)), [map_teams[team] for team in data["home_team"]]] = 1
    return goals_vector, tau_home


@decorators.verify_required_column(column_names=["away_team", "ftag"])
def compute_goals_away_vectors(
    data: pd.DataFrame, /, map_teams: dict[str, int], nbr_team: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute vectors representing away team goals.

    Args:
        data (pd.DataFrame): Input DataFrame with away team goals and AwayTeam column.
        map_teams (dict): Dictionary mapping team names to numerical IDs.
        nbr_team (int): Number of teams in the league.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            goals_vector representing away team goals and tau_away representing binary vectors
            for each away team.

    """
    goals_vector = data["ftag"].values
    tau_away = np.zeros((len(data), nbr_team))
    tau_away[np.arange(len(data)), [map_teams[team] for team in data["away_team"]]] = 1
    return goals_vector, tau_away


def poisson_proba(lambda_param: float, k: int) -> np.ndarray:
    """Calculate the probability of achieving up to k goals given a lambda parameter.

    Args:
        lambda_param (float): The expected number of goals.
        k (int): Number of goal values to return, starting at zero.

    Returns:
        np.ndarray:  An array containing the probabilities of achieving each possible
            number of goals from 0 through ``k - 1``.

    """
    poisson = stats.poisson(mu=lambda_param)
    k_list = np.arange(k)
    return poisson.pmf(k=k_list)  # type:ignore


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
    proba_draw = bookmaker_proba[1]
    proba_home_win = bookmaker_proba[0]

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
