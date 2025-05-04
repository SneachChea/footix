import itertools
import math

import numpy as np
from scipy.stats import skellam

from footix.strategy.bets import Bet


def generate_combinations(selections: list[Bet]) -> tuple[list[list[int]], list[float]]:
    """Generate a matrix of all possible combinations of selections and their corresponding
    probabilities.

    Args:
        selections (list[Bet]):
            A list of Bet object representing the selectable options,
    Returns:
        tuple[list[list[int]], list[float]]: A tuple containing two lists:
            1. A list of lists, where each sublist represents a combination of selections (0 or 1),
            indicating which options are selected in that combination.
            2. A list of probabilities corresponding to each combination.

    """
    combinations = []
    probs = []

    for c in range(len(selections) + 1):
        for subset in itertools.combinations(selections, c):
            combination = [1 if selection in subset else 0 for selection in selections]
            prob = 1.0
            for bet in selections:
                prob *= bet.prob_mean if bet in subset else 1 - bet.prob_mean
            combinations.append(combination)
            probs.append(prob)
    return combinations, probs


def generate_bets_combination(
    selections: list[Bet], max_multiple: int
) -> tuple[list[list[int]], list[float]]:
    """Generates all possible bets based on selections and a maximum multiple.

    Args:
        selections (list[dict]): A list of dictionaries, where each dictionary contains selection
        information, including the "odds_book" key for the odds in the book.
        max_multiple (int): The maximum number of selections that can be combined in a strategy.

    Returns:
        tuple[list[list[int]], list[float]]: The first list contains all possible bets, where each
            bet is represented as a list of 1s and 0s indicating the selection. The second list
            contains the product of odds for each combination, representing the book odds.

    """
    bets = []
    book_odds = []

    for multiple in range(1, max_multiple + 1):
        for subset in itertools.combinations(selections, multiple):
            bet = [1 if selection in subset else 0 for selection in selections]
            prod = 1.00
            for selection in subset:
                prod *= selection.odds
            bets.append(bet)
            book_odds.append(prod)

    return bets, book_odds


def compute_stacks(
    stakes: list[float],
    bankroll: float,
    combinations: list[list[int]],
    winning_bets: dict[int, list[int]],
    book_odds: list[float],
    probs,
    eps: float = 1e-9,
):
    """Compute the expected bankroll after placing bets.

    Args:
        stakes (list[float]): The amount of money placed on each bet.
        bankroll (float): The initial amount of money available.
        combinations (list[list[int]]): A list of combinations of bet indices.
        winning_bets (dict[int, list[int]]): A dictionary where keys are bet indices and
        values are lists of combination indices that win.
        book_odds (list[float]): The odds provided by the bookmaker for each bet.
        probs (list[float]): The probabilities of each combination occurring.
        eps (float, optional): A small value to avoid log(0). Defaults to 1e-9.

    Returns:
        float: The negative sum of the expected log bankrolls.

    """

    end_bankrolls = np.array([bankroll - np.sum(stakes)] * len(combinations), dtype=float)
    for index_bet, comb_indices in winning_bets.items():
        for index in comb_indices:
            end_bankrolls[index] += stakes[index_bet] * book_odds[index_bet]
    # Avoid log(0) by adding a small epsilon.
    return -np.sum([p * math.log(max(e, eps)) for p, e in zip(probs, end_bankrolls)])


def _skellam_post_probs(
    lh: np.ndarray, la: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Probabilités a posteriori (vecteurs) : home‑win, draw, away‑win pour des échantillons
    λ_home, λ_away de même longueur."""
    p_home = 1 - skellam.cdf(0, lh, la)  # P(diff > 0)
    p_draw = skellam.pmf(0, lh, la)  # P(diff = 0)
    p_away = skellam.cdf(-1, lh, la)  # P(diff < 0)
    return p_home, p_draw, p_away
