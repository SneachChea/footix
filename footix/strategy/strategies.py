import itertools
import math
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from tqdm.auto import tqdm

import footix.utils.decorators as decorators


@decorators.verify_required_column(
    ["Home_Team", "Away_Team", "C_H", "C_D", "C_A", "P_H", "P_D", "P_A"]
)
def classic_kelly(input_df: pd.DataFrame, bankroll: float) -> None:
    """Classic Kelly criterion function.

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
    num_iterations: int = 1000,
    learning_rate: float = 0.1,
    penalty_weight: float = 1000.0,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    early_stopping: bool = True,
    tolerance: int = 5,
) -> list[dict[str, Any]]:
    """Compute the real Kelly criterion using a GPU accelerated gradient-based optimizer
    (PyTorch).

    Args:
        selections (list[dict[str, Any]]): List of betting selections.
        bankroll (float): Total bankroll available.
        max_multiple (int, optional): Maximum number of selections to combine. Defaults to 1.
        num_iterations (int, optional): Number of iterations for gradient descent.
        learning_rate (float, optional): Learning rate for the optimizer.
        penalty_weight (float, optional): Weight for the penalty term enforcing the bankroll
            constraint.
        device (str, optional): Device to run the computations on ("cuda" or "cpu").
        early_stopping (bool, optional): Whether to stop early if convergence is detected.
        tolerance (int, optional): Tolerance for early stopping.

    Returns:
        list[dict]: A dictionary containing, for each bet with non-negligible stake,
              the bet string, the bet odd, and the stake.

    """
    start_time = time.time()

    if max_multiple > len(selections):
        raise ValueError(f"Error: max_multiple must not exceed {len(selections)}")

    # Generate combinations and bets (this part runs on CPU)
    combinations, probs = generate_combinations(selections)
    bets, book_odds = generate_bets_combination(selections, max_multiple)

    # Build winning_bets mapping
    winning_bets: dict[int, list[int]] = defaultdict(list)
    for index_combination, combination in enumerate(combinations):
        for index_bet, bet in enumerate(bets):
            if sum(c * b for c, b in zip(combination, bet)) == sum(bet):
                winning_bets[index_bet].append(index_combination)

    # Convert constant lists to torch tensors (on the chosen device)
    probs_t = torch.tensor(probs, device=device, dtype=torch.float32)
    book_odds_t = torch.tensor(book_odds, device=device, dtype=torch.float32)
    bankroll_t = torch.tensor(bankroll, device=device, dtype=torch.float32)
    eps = 1e-9

    # Number of bets (decision variables)
    num_bets = len(bets)

    # Initialize stakes as torch parameters (starting with an equal fraction of bankroll)
    stakes = torch.nn.Parameter(torch.full((num_bets,), bankroll / (2 * num_bets), device=device))

    optimizer = Adam([stakes], lr=learning_rate)

    # Precompute a mapping from each bet to the list of outcome indices that are won.
    # For GPU efficiency, we create a mask matrix of shape (num_bets, num_outcomes).
    num_outcomes = len(combinations)
    win_mask = torch.zeros((num_bets, num_outcomes), device=device)
    for bet_idx, outcome_indices in winning_bets.items():
        win_mask[bet_idx, outcome_indices] = 1.0

    old_loss = torch.tensor(torch.inf, device=device)
    counter = 0
    # Optimization loop
    with tqdm(total=num_iterations) as pbar:
        for iter in range(num_iterations):
            optimizer.zero_grad()

            # Compute base bankroll remaining after placing all stakes.
            total_stake = torch.sum(stakes)
            base = bankroll_t - total_stake

            # Create an outcome vector: for each outcome, add winnings from bets that win.
            # stakes: (num_bets,), win_mask: (num_bets, num_outcomes), book_odds_t: (num_bets,)
            # Compute winnings per outcome as a sum over bets:
            winnings = torch.matmul((stakes * book_odds_t), win_mask)  # shape: (num_outcomes,)
            end_bankrolls = base + winnings

            # Compute the objective: negative weighted sum of log(end_bankrolls)
            # Add eps to avoid log(0).
            objective = -torch.sum(probs_t * torch.log(end_bankrolls + eps))

            # Penalty to enforce the bankroll constraint (if total_stake exceeds bankroll)
            constraint_violation = torch.clamp(total_stake - bankroll_t, min=0.0)
            penalty = penalty_weight * (constraint_violation**2)

            loss = objective + penalty

            loss.backward()
            optimizer.step()
            # Clamp stakes to be non-negative
            pbar.set_postfix(loss=f"{loss.item():.5f}", stake=f"{total_stake.item():.2f}")
            pbar.update(1)

            with torch.no_grad():
                stakes.clamp_(min=0.0)
                if early_stopping:
                    if torch.isclose(old_loss, loss, rtol=1e-7):
                        if counter == tolerance:
                            break
                        else:
                            counter += 1
            old_loss = loss

    runtime = time.time() - start_time
    print(
        f"\n{datetime.now().replace(microsecond=0)}"
        f"- Optimization finished. Runtime --- "
        f"{round(runtime, 3)} seconds ---\n"
    )
    final_objective = loss.item()
    print(f"Objective: {round(final_objective, 5)}")
    ce = math.exp(-final_objective)
    print(f"Certainty Equivalent: {round(ce, 3)}\n")

    # Collect and display the bets with non-negligible stakes
    results = []
    sum_stake = 0
    stakes_final = stakes.detach().cpu().numpy()
    for index_bet, bet in enumerate(bets):
        bet_strings = [
            selections[index_sel]["name"] for index_sel, sel in enumerate(bet) if sel == 1
        ]
        stake_value = stakes_final[index_bet]
        if stake_value >= 0.50:
            bet_string = " / ".join(bet_strings)
            odd = round(book_odds[index_bet], 3)
            print(f"{bet_string} @ {odd}" f"- € {int(round(stake_value, 0))}")
            results.append({"match": bet_string, "odd": odd, "stake": int(round(stake_value, 0))})
            sum_stake += stake_value
    print(f"Bankroll used: {sum_stake:.2f} €")
    return results


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
    """Convert an index to a result string.

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
    """Generate a matrix of all possible combinations of selections and their corresponding
    probabilities.

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
                prod *= selection["odds_bookie"]
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
