import itertools

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
