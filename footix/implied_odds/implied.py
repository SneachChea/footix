import operator
from typing import cast

import numpy as np
from scipy import optimize

from footix.utils.typing import ArrayLikeF

# Most of those functions are inspired by the awesome package penaltyblog
# https://github.com/martineastwood/penaltyblog/tree/master


def _assert_odds(odds: ArrayLikeF, axis: None | int = None) -> None:
    if (not isinstance(odds, list)) and (not isinstance(odds, np.ndarray)):
        raise TypeError("Odds must be a list or an numpy array.")
    if isinstance(odds, list):
        odds = np.array(odds)
    if axis is not None:
        if odds.shape[axis] != 3:
            raise ValueError("It is a football package ! You must provide 3 odds.")
    else:
        if odds.shape[0] != 3:
            raise ValueError("It is a football package ! You must provide 3 odds.")
    if (odds < 1.0).any():
        raise ValueError("All odds must be greater then 1.")


def multiplicative(odds: ArrayLikeF, axis: int = -1) -> tuple[np.ndarray, float | np.ndarray]:
    """Multiplicative way to normalize the odds. Work for multidimensionnal array.

    Args:
        odds (list or np.array): list of odds
        axis (int) : axis where compute the probabilities

    """
    _assert_odds(odds, axis=axis)
    if isinstance(odds, list):
        odds = np.array(odds)
    if len(odds.shape) > 1:
        normalization = np.sum(1.0 / odds, axis=axis, keepdims=True)
    else:
        normalization = np.sum(1.0 / odds, axis=axis)
    margin = normalization - 1.0
    return 1.0 / (normalization * odds), margin


def power(odds: ArrayLikeF) -> tuple[np.ndarray, float]:
    """From penaltyblog package. The power method computes the implied probabilities by solving
    for the power coefficient that normalizes the inverse of the odds to sum to 1.0.

    Args:
        odds : (list or np.array): list of odds

    """
    _assert_odds(odds)
    if isinstance(odds, list):
        odds = np.array(odds)
    inv_odds = 1.0 / odds
    margin = cast(float, np.sum(inv_odds) - 1.0)

    def _fit(k: float, inv_odds: np.ndarray) -> float:
        implied = operator.pow(inv_odds, k)
        return 1 - np.sum(implied)

    res = optimize.ridder(_fit, 0, 100, args=(inv_odds,))
    normalized = operator.pow(inv_odds, res)
    return normalized, margin


def shin(odds: ArrayLikeF) -> tuple[np.ndarray, float]:
    """Computes the implied probabilities via Shin's method (1992, 1993).

    Args:
        odds (list or np.ndarray): An array of size 3 containing the odds for Home victory, draw,
                                   or Away victory, respectively.

    Returns:
        tuple:
            - np.ndarray: The implied probabilities for each outcome.
            - float: The margin.

    """
    _assert_odds(odds)

    if isinstance(odds, list):
        odds = np.array(odds)

    inv_odds = 1.0 / odds
    margin = cast(float, np.sum(inv_odds) - 1.0)

    def _fit(z_param: float, inv_odds: np.ndarray) -> float:
        implied = _shin(z_param, inv_odds)
        return 1.0 - np.sum(implied)

    res = optimize.ridder(_fit, 0, 100, args=(inv_odds,))
    normalized = _shin(res, inv_odds)
    return normalized, margin


def _shin(z_param: float, inv_odds: np.ndarray) -> np.ndarray:
    """Computes the implied probabilities using Shin's method.

    Args:
        z_param (float): The Shin adjustment parameter.
        inv_odds (np.ndarray): An array of size 3 containing the inverse odds for Home victory,
                               draw, or Away victory, respectively.

    Returns:
        np.ndarray: The implied probabilities for each outcome.

    """
    normalized = np.sum(inv_odds)
    implied = (
        np.sqrt(z_param**2 + 4 * (1 - z_param) * inv_odds**2 / normalized) - z_param
    ) / (2 - 2 * z_param)
    return implied
