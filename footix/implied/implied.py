import numpy as np
from scipy import optimize

__all__ = ["multiplicative", "power", "shin"]
# Most of those functions come from the awesome package penaltyblog


def multiplicative(
    odds: list | np.ndarray, axis: int = -1
) -> tuple[np.ndarray, float | np.ndarray]:
    """
        multiplicative way to normalize the odds.

    Args:
        odds (list or np.array): list of odds
        axis (int) : axis where compute the probabilities
    """
    if isinstance(odds, list):
        odds = np.array(odds)
    if len(odds.shape) > 1:
        Z = np.sum(1.0 / odds, axis=axis, keepdims=True)
    else:
        Z = np.sum(1.0 / odds, axis=axis)
    margin = Z - 1.0
    return 1.0 / (Z * odds), margin


def power(odds: list | np.ndarray) -> tuple:
    """
    From penaltyblog package.
    The power method computes the implied probabilities by solving for the
    power coefficient that normalizes the inverse of the odds to sum to 1.0

    Args:
        odds : (list or np.array): list of odds
    """
    if isinstance(odds, list):
        odds = np.array(odds)
    inv_odds = 1.0 / odds
    margin = np.sum(inv_odds) - 1.0

    def _power(k, inv_odds):
        implied = inv_odds**k
        return implied

    def _power_error(k, inv_odds):
        implied = _power(k, inv_odds)
        return 1 - np.sum(implied)

    res = optimize.ridder(_power_error, 0, 100, args=(inv_odds,))
    normalized = _power(res, inv_odds)

    return normalized, margin


def shin(odds: list | np.ndarray) -> tuple:
    """
    Computes the implied probabilities via the Shin (1992, 1993) method

    Args:
        odds : (list or np.ndarray): list of odds
    """
    if isinstance(odds, list):
        odds = np.array(odds)
    inv_odds = 1.0 / odds
    margin = np.sum(inv_odds) - 1.0

    def _shin_error(z, inv_odds):
        implied = _shin(z, inv_odds)
        return 1 - np.sum(implied)

    def _shin(z, inv_odds):
        implied = ((z**2 + 4 * (1 - z) * inv_odds**2 / np.sum(inv_odds)) ** 0.5 - z) / (
            2 - 2 * z
        )
        return implied

    res = optimize.ridder(_shin_error, 0, 100, args=(inv_odds,))
    normalized = _shin(res, inv_odds)
    return normalized, margin
