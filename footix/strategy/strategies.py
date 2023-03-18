import math
import time
import itertools
import numpy as np
import pandas as pd
import scipy.optimize
from datetime import datetime
from collections import defaultdict
from typing import List, Dict

def realKelly(selections: List[Dict], bankroll: float, max_multiple: int = 1) -> None:
    """
        Compute the real Kelly criterion for mutually exclusive bets. 
        This function comes from https://github.com/BettingIsCool/real_kelly-independent_concurrent_outcomes-/blob/master/real_kelly-independent_concurrent_outcomes-.py

    Args:
        selections (List[Dict]): selections of bets. This arguments is a list of dictionnary with keys 'name','odds_bookie',
        'probability'
        bankroll (float): the bankroll
        max_multiple (int, optional): max length for combined bets. Defaults to 1.

    Raises:
        ValueError: max_multiple must not exceed the number of bets

    Returns:
        None
    """

    start_time = time.time()

    # MAXIMUM TEAMS IN A MULTIPLE MUST NOT EXCEED LEN(SELECTIONS)
    if max_multiple > len(selections):
        raise ValueError(f'Error: Maximum multiple must not exceed {len(selections)}')

    # CREATE A MATRIX OF POSSIBLE COMBINATIONS AND A PROBABILITY VECTOR OF SIZE LEN(COMBINATIONS)
    combinations = []
    probs = []

    for c in range(0, len(selections) + 1):
        for subset in itertools.combinations(selections, c):
            combination, prob = list(), 1.00
            for selection in selections:
                if selection in subset:
                    combination.append(1)
                    prob *= selection['probability']
                else:
                    combination.append(0)
                    prob *= (1 - selection['probability'])
            combinations.append(combination)
            probs.append(prob)

    # CREATE A MATRIX OF POSSIBLE SINGLES & MULTIPLES
    bets = []
    book_odds = []

    for multiple in range(1, max_multiple + 1):
        for subset in itertools.combinations(selections, multiple):
            bet, prod = list(), 1.00
            for selection in selections:
                if selection in subset:
                    bet.append(1)
                    prod *= selection['odds_book']
                else:
                    bet.append(0)
            bets.append(bet)
            book_odds.append(prod)

    # CACHE WINNING BETS
    winning_bets = defaultdict(list)
    for index_combination, combination in enumerate(combinations):
        for index_bet, bet in enumerate(bets):
            if sum([c * b for c, b in zip(combination, bet)]) == sum(bet):
                winning_bets[index_bet].append(index_combination)

    def f(stakes):
        """ This function will be called by scipy.optimize.minimize repeatedly to find the global maximum """

        # INITIALIZE END_BANKROLLS AND OBJECTIVE BEFORE EACH OPTIMIZATION STEP
        end_bankrolls = len(combinations) * [bankroll - np.sum(stakes)]

        for index_bet, index_combinations in winning_bets.items():
            for index_combination in index_combinations:
                end_bankrolls[index_combination] += stakes[index_bet] * book_odds[index_bet]

        # RETURN THE OBJECTIVE AS A SUMPRODUCT OF PROBABILITIES AND END_BANKROLLS - THIS IS THE FUNCTION TO BE MAXIMIZED
        return -sum([p * e for p, e in zip(probs, np.log(end_bankrolls))])

    def constraint(stakes):
        """ Sum of all stakes must not exceed bankroll """
        return sum(stakes)

    # FIND THE GLOBAL MAXIMUM USING SCIPY'S CONSTRAINED MINIMIZATION
    bounds = list(zip(len(bets) * [0], len(bets) * [bankroll]))
    nlc = scipy.optimize.NonlinearConstraint(constraint, -np.inf, bankroll)
    res = scipy.optimize.differential_evolution(func=f, bounds=bounds, constraints=(nlc))

    runtime = time.time() - start_time
    print(f"\n{datetime.now().replace(microsecond=0)} - Optimization finished. Runtime --- {round(runtime, 3)} seconds ---\n")
    print(f"Objective: {round(res.fun, 5)}")
    print(f"Certainty Equivalent: {round(math.exp(-res.fun), 3)}\n")

    # CONSOLE OUTPUT
    sum_stake = 0
    for index_bet, bet in enumerate(bets):
        bet_strings = list()
        for index_sel, sel in enumerate(bet):
            if sel == 1:
                bet_strings.append(selections[index_sel]['name'])

        stake = res.x[index_bet]
        if stake >= 0.50:
            print(f"{(' / ').join(bet_strings)} @{round(book_odds[index_bet], 3)} - € {int(round(stake, 0))}")
            sum_stake+=stake
    print(f"Bankroll used {sum_stake} €")

def selectBets(odds_bookie: pd.DataFrame, probas: np.ndarray) -> List[Dict]:
    """Select bets profitable in the sense p > 1./o


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
            if probas[idx, i]> 1./odd_object[i]:
                selections.append({"name": fromIdx2Res(i, rows["Home team"], rows["Away team"]), "odds_book": odd_object[i],
                "probability": probas[idx, i]})
    return selections
                


def fromIdx2Res(index: int, HomeTeam: str, AwayTeam: str)->str:
    if index == 0:
        return f"Victoire à domicile de {HomeTeam} (contre {AwayTeam})"
    if index == 1:
        return f"Match nul ({HomeTeam} vs {AwayTeam})"
    return  f"Victoire à l'extérieur de {AwayTeam} (contre {HomeTeam})"