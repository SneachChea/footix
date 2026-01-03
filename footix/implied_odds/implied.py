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


def multiplicative_method(
    odds: ArrayLikeF, axis: int = -1
) -> tuple[np.ndarray, float | np.ndarray]:
    """Multiplicative way to normalize the odds. Work for multidimensionnal array.

    Args:
        odds (list or np.array): list of odds
        axis (int) : axis where compute the probabilities

    """
    _assert_odds(odds, axis=axis)
    odds = np.asarray(odds, dtype=float)
    if len(odds.shape) > 1:
        normalization = np.sum(1.0 / odds, axis=axis, keepdims=True)
    else:
        normalization = np.sum(1.0 / odds, axis=axis)
    margin = normalization - 1.0
    return 1.0 / (normalization * odds), margin


def power_method(
    odds: ArrayLikeF, *, max_iter: int = 50, tol: float = 1e-6
) -> tuple[np.ndarray, float]:
    """Compute implied probabilities using the power–margin method.

    This function takes a collection of decimal (European) odds and returns a vector of
    implied probabilities that sum to one, while also computing the bookmaker’s margin.
    The power–margin approach raises each inverse-odds entry to a common exponent (k) that
    exactly normalizes them. Numerically, we solve for k via a Newton iteration in log-space.

    Parameters
    ----------
    odds : ArrayLike
        A one-dimensional array-like of positive decimal odds. Each entry must be strictly
        greater than zero.
    tol : float, default=1e-12
        Convergence tolerance for the root-finding procedure. The iteration stops when
        |∑(1/odds)**k − 1| < tol.
    max_iter : int, default=50
        Maximum number of Newton steps to attempt. If convergence is not reached within
        this many iterations, a RuntimeError is raised.

    Returns
    -------
    probs : np.ndarray
        A 1-D array of implied probabilities corresponding to each input odd. These probabilities
        are non-negative and sum exactly (to machine precision) to 1.0.
    margin : float
        The bookmaker’s over-round (or “vigorish”), computed as ∑(1/odds) − 1. A value of zero
        indicates a fair book (no margin).

    Raises
    ------
    ValueError
        If `odds` is not a one-dimensional array-like, or if any entry in `odds` is ≤ 0.
    RuntimeError
        If the Newton root-finder fails to converge within `max_iter` iterations.

    Notes
    -----
    1. When `margin` is already within `tol` of zero, the function treats the book as fair and
        returns the normalized inverses of the odds directly.
    2. Internally, we solve
           f(k) = Σ (1/odds_i)**k − 1 = 0
       by applying Newton’s method to the equivalent form
           f(k) = Σ exp(k * log(1/odds_i)) − 1.
       Working in log-space improves numerical stability, especially when odds are large
       (inv-odds small).
    3. The default initial guess for k is 1. For typical sportsbook margins (up to 10–15%),
        convergence is very fast—often under 5 iterations.

    Examples
    --------
    >>> odds = [1.80, 2.10, 4.00]
    >>> probs, margin = implied_probs_power(odds)
    >>> np.isclose(probs.sum(), 1.0)
    True
    >>> margin  # e.g., around 0.043 (4.3% over-round)
    0.043

    """
    _assert_odds(odds)
    odds = np.asarray(odds, dtype=float)
    inv_odds = 1.0 / odds
    margin = cast(float, np.sum(inv_odds) - 1.0)
    log_inv = np.log(inv_odds)

    def f(k: float) -> float:
        return np.exp(k * log_inv).sum() - 1.0

    def fprime(k: float) -> float:
        y = np.exp(k * log_inv)
        return (y * log_inv).sum()

    k = 1.0
    for _ in range(max_iter):
        fk = f(k)
        if abs(fk) < tol:
            break
        k -= fk / fprime(k)
    else:
        raise RuntimeError("Power root-finder did not converge.")

    probs = np.exp(k * log_inv)
    return probs / probs.sum(), margin


def shin_method(odds: ArrayLikeF, *, tol: float = 1e-12) -> tuple[np.ndarray, float]:
    """Compute implied probabilities and bookmaker margin using Shin’s method.

    Shin’s method (Shin, 1992; Shin, 1993) adjusts raw decimal odds for insider‐information
    risk by finding a parameter z in (0, 1) that forces the “Shin‐adjusted” probabilities
    to sum to 1. This implementation uses Brent’s root‐finding algorithm to solve for z.

    Parameters
    ----------
    odds : array‐like of float, shape (3,)
        Decimal odds for the three mutually exclusive outcomes, in the order:
        [home_win, draw, away_win]. Each entry must be strictly positive.
    tol : float, optional
        Absolute tolerance for the Brent solver when finding the Shin parameter z.
        Default is 1e‐12.

    Returns
    -------
    implied : ndarray, shape (3,)
        Shin‐adjusted probabilities for [home_win, draw, away_win]. These probabilities
        account for bookmaker over‐round and the presence of insider information, and they
        sum to 1 within numerical tolerance.
    margin : float
        Bookmaker over‐round (also called “vig” or “juice”), computed as
            margin = sum(1 / odds_i) − 1.

    Raises
    ------
    ValueError
        If `odds` does not have exactly three elements or if any element is non‐positive.

    Notes
    -----
    1. Let q_i = 1 / odds_i and Q = sum(q_i). For a given z in (0, 1), Shin’s formula gives:
         p_i(z) = ( sqrt(z^2 + 4 (1 − z) q_i^2 / Q) − z ) / [2 (1 − z)].
       The root‐finding problem is:
         f(z) = sum_i p_i(z) − 1 = 0.
       We bracket z within (ε, 1 − ε) to avoid division by zero (ε ≈ 1e‐12).

    2. Once z is found, the implied probabilities p_i(z) automatically sum to 1 (within tol).

    """

    _assert_odds(odds)
    odds_arr = np.asarray(odds)
    inv_odds = 1.0 / odds_arr
    margin: float = cast(float, inv_odds.sum() - 1.0)

    inv_sq = inv_odds**2
    total_inv = inv_odds.sum()

    def _objective(z: float) -> float:
        """Equation whose root forces the implied probabilities to sum to 1."""
        # Equation (8) in Shin (1992):
        root_term = np.sqrt(z * z + 4.0 * (1.0 - z) * inv_sq / total_inv)
        prob_sum = ((root_term - z) / (2.0 * (1.0 - z))).sum()
        return prob_sum - 1.0  # zero at the correct *z*

    # In theory 0 < z < 1; shrink the bracket slightly to avoid division errors.
    z_star = optimize.brentq(_objective, 1e-12, 1.0 - 1e-12, xtol=tol)
    implied = _shin_probabilities(inv_odds, z_star)  # type: ignore
    return implied, margin


def _shin_probabilities(inv_odds: np.ndarray, z: float) -> np.ndarray:
    """Vectorised Shin probability transform."""
    total_inv = inv_odds.sum()
    root_term = np.sqrt(z * z + 4.0 * (1.0 - z) * inv_odds**2 / total_inv)
    return (root_term - z) / (2.0 * (1.0 - z))
