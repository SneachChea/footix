import operator
from typing import cast

import numpy as np
from scipy import optimize

# Most of those functions are inspired by the awesome package penaltyblog
# https://github.com/martineastwood/penaltyblog/tree/master


def _assert_odds(odds: list[float] | np.ndarray, axis: None | int = None) -> None:
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


def multiplicative(
    odds: list[float] | np.ndarray, axis: int = -1
) -> tuple[np.ndarray, float | np.ndarray]:
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


def power(odds: list[float] | np.ndarray) -> tuple[np.ndarray, float]:
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


def shin(odds: list[float] | np.ndarray) -> tuple[np.ndarray, float]:
    """Computes the implied probabilities via the Shin (1992, 1993) method.

    Args:
        odds : (list or np.ndarray): list of odds

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
    """Compute the implied probability using Shin's method."""
    normalized = np.sum(inv_odds)
    implied = (
        np.sqrt(z_param**2 + 4 * (1 - z_param) * inv_odds**2 / normalized) - z_param
    ) / (2 - 2 * z_param)
    return implied
