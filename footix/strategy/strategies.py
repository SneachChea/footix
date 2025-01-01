import itertools
import math
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import scipy.optimize

import footix.utils.decorators as decorators


@decorators.verify_required_column(
    ["Home_Team", "Away_Team", "C_H", "C_D", "C_A", "P_H", "P_D", "P_A"]
)
def classic_kelly(input_df: pd.DataFrame, bankroll: float) -> None:
    """
    Classic Kelly criterion function.
    Parameters
    ----------
    input_df: pandas.DataFrame
    bankroll: float
        The current bankroll.
    Returns
    -------
    None
        Modifies the input DataFrame in place.
    """

    def _kelly_criterion(odds: pd.Series, probability: pd.Series, bankroll: float) -> pd.Series:
        kelly = bankroll * (probability * (odds - 1.0) - 1.0 + probability) / (odds - 1.0)
        kelly[kelly < 0.0] = 0.0
        return kelly

    input_df["Kelly_H"] = _kelly_criterion(input_df["C_H"], input_df["P_H"], bankroll)
    input_df["Kelly_A"] = _kelly_criterion(input_df["C_A"], input_df["P_A"], bankroll)
    input_df["Kelly_D"] = _kelly_criterion(input_df["C_D"], input_df["P_D"], bankroll)


def realKelly(
    selections: list[dict[str, Any]],
    bankroll: float,
    max_multiple: int = 1,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> None:
    """
        Compute the real Kelly criterion for mutually exclusive bets.
        This function comes from
        https://github.com/BettingIsCool/real_kelly-independent_concurrent_outcomes-/blob/master/
        real_kelly-independent_concurrent_outcomes-.py

    Args:
        selections (List[dict[str, Any]]): selections of bets. This arguments is a
        list of dictionnary with keys 'name','odds_bookie',
        'probability'
        bankroll (float): the bankroll
        max_multiple (int, optional): max length for combined bets. Defaults to 1.

    Raises:
        ValueError: max_multiple must not exceed the number of bets

    Returns:
        None
    """

    start_time = time.time()

    if max_multiple > len(selections):
        raise ValueError(f"Error: Maximum multiple must not exceed {len(selections)}")

    combinations, probs = generate_combinations(selections)
    bets, book_odds = generate_bets_combination(selections, max_multiple)

    winning_bets: defaultdict[int, list[int]] = defaultdict(list)
    for index_combination, combination in enumerate(combinations):
        # Iterate over each bet to check if the sum of combinations equals the sum of the bet
        for index_bet, bet in enumerate(bets):
            if sum(c * b for c, b in zip(combination, bet)) == sum(bet):
                # If true, add the combination index to the list of winning bets for this bet index
                winning_bets[index_bet].append(index_combination)

    def constraint(stakes):
        """Sum of all stakes must not exceed bankroll"""
        return sum(stakes)

    # FIND THE GLOBAL MAXIMUM USING SCIPY'S CONSTRAINED MINIMIZATION
    bounds = list(zip(len(bets) * [0], len(bets) * [bankroll]))
    nlc = scipy.optimize.NonlinearConstraint(constraint, 0.0, bankroll)
    res = scipy.optimize.differential_evolution(
        func=compute_stacks,
        bounds=bounds,
        constraints=(nlc),
        args=(
            bankroll,
            combinations,
            winning_bets,
            book_odds,
            probs,
        ),
        **(optimizer_kwargs or {}),
    )

    runtime = time.time() - start_time
    print(
        f"\n{datetime.now().replace(microsecond=0)} - Optimization finished. Runtime",
        f"--- {round(runtime, 3)} seconds ---\n",
    )
    print(f"Objective: {round(res.fun, 5)}")
    print(f"Certainty Equivalent: {round(math.exp(-res.fun), 3)}\n")

    # CONSOLE OUTPUT
    sum_stake = 0
    for index_bet, bet in enumerate(bets):
        bet_strings = list()
        for index_sel, sel in enumerate(bet):
            if sel == 1:
                bet_strings.append(selections[index_sel]["name"])

        stake = res.x[index_bet]
        if stake >= 0.50:
            print(
                f"{(' / ').join(bet_strings)} @{round(book_odds[index_bet], 3)}",
                f"- € {int(round(stake, 0))}",
            )
            sum_stake += stake
    print(f"Bankroll used {sum_stake} €")


@decorators.verify_required_column(column_names={"1", "2", "N"})
def selectBets(odds_bookie: pd.DataFrame, probas: np.ndarray) -> list[dict]:
    """
    Select bets profitable in the sense p > 1./o
    Args:
        odds_bookie (pd.DataFrame): odds from fdj scrapper
        probas (np.ndarray): probability from the custom model. Size (len(matchs), 3)

    Returns:
        List[Dict]: a list of selected bets with the syntax adapted to realKelly
    """
    selections = []
    for idx, rows in odds_bookie.iterrows():
        odd_object = rows[["1", "N", "2"]].to_numpy()
        for i in range(3):
            if probas[idx, i] > 1.0 / odd_object[i]:
                selections.append(
                    {
                        "name": _fromIdx2Res(i, rows["Home team"], rows["Away team"]),
                        "odds_book": odd_object[i],
                        "probability": probas[idx, i],
                    }
                )
    return selections


def _fromIdx2Res(index: int, HomeTeam: str, AwayTeam: str) -> str:
    """
    Convert an index to a result string.

    Args:
        index (int): Index of the result.
        HomeTeam (str): Name of the home team.
        AwayTeam (str): Name of the away team.

    Returns:
        str: A string describing the result
    """
    if index == 0:
        return f"Victoire à domicile de {HomeTeam} contre {AwayTeam}"
    elif index == 1:
        return f"Match nul ({HomeTeam} vs {AwayTeam})"
    else:
        return f"Victoire à l'extérieur de {AwayTeam} contre {HomeTeam}"


def generate_combinations(selections: list[dict[str, Any]]) -> tuple[list[list[int]], list[float]]:
    """Generate a matrix of all possible combinations of selections
        and their corresponding probabilities.

    Args:
        selections (list[dict[str, Any]]):
                A list of dictionaries representing the selectable options,
                where each dictionary contains
            - 'probability': The probability associated with selecting that option.

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
            for selection in selections:
                prob *= (
                    selection["probability"]
                    if selection in subset
                    else 1 - selection["probability"]
                )
            combinations.append(combination)
            probs.append(prob)
    return combinations, probs


def generate_bets_combination(
    selections: list[dict], max_multiple: int
) -> tuple[list[list[int]], list[float]]:
    """
    Generates all possible bets based on selections and a maximum multiple.

    Parameters:
    selections (list[dict]):
            A list of dictionaries, where each dictionary contains selection information,
            including the "odds_book" key for the odds in the book.
    max_multiple (int): The maximum number of selections that can be combined in a strategy.

    Returns:
    tuple[list[list[int]], list[float]]:
                A tuple containing two lists. The first list contains all possible bets,
                where each bet is represented as a list of 1s and 0s indicating the selection.
                The second list contains the product of odds for each combination,
                representing the book odds.
    """
    bets = []
    book_odds = []

    for multiple in range(1, max_multiple + 1):
        for subset in itertools.combinations(selections, multiple):
            bet = [1 if selection in subset else 0 for selection in selections]
            prod = 1.00
            for selection in subset:
                prod *= selection["odds_bookie"]
            bets.append(bet)
            book_odds.append(prod)

    return bets, book_odds


def compute_stacks(
    stakes,
    bankroll: float,
    combinations: list[list[int]],
    winning_bets: dict[int, list[int]],
    book_odds: list[float],
    probs,
):
    """
    This function will be called by scipy.optimize.minimize repeatedly to
    find the global maximum
    """
    end_bankrolls = len(combinations) * [bankroll - np.sum(stakes)]

    for index_bet, index_combinations in winning_bets.items():
        for index_combination in index_combinations:
            end_bankrolls[index_combination] += stakes[index_bet] * book_odds[index_bet]

    return -sum([p * e for p, e in zip(probs, np.log(end_bankrolls))])
