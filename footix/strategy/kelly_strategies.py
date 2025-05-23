import math
import time
from collections import defaultdict
from datetime import datetime
from typing import Literal

import numpy as np
import torch
from torch.optim import Adam
from tqdm.auto import tqdm

import footix.strategy._utils as strat_utils
from footix.strategy.bets import Bet


def classic_kelly(list_bets: list[Bet], bankroll: float) -> list[Bet]:
    """Compute bet stakes based on the classic Kelly criterion.

    This function calculates the stake for each bet in the input list using the traditional Kelly
    formula. For each bet, it determines the Kelly fraction as the ratio of the estimated edge
    (i.e., the advantage) over the net odds (odds - 1). Negative Kelly fractions are set to zero,
    ensuring no bet is made when the edge is unfavorable. The final stake for each bet is computed
    by multiplying the Kelly fraction by the available bankroll.

    Args:
        list_bets (list[Bet]): A list of Bet objects. Each Bet must have an 'odds' attribute
                                representing the betting odds and an 'edge_mean' attribute
                                indicating the estimated edge.
        bankroll (float): The total available bankroll to allocate across the bets.

    Returns:
        list[Bet]: The list of Bet objects with their 'stake' attribute updated according to
                    the Kelly criterion.

    """

    for b in list_bets:
        edge = b.edge_mean
        odd = b.odds
        kelly_val = edge / (odd - 1.0)
        b.stake = bankroll * max(kelly_val, 0)
    return list_bets


def realKelly(
    list_bets: list[Bet],
    bankroll: float,
    max_multiple: int = 1,
    num_iterations: int = 1000,
    learning_rate: float = 0.1,
    penalty_weight: float = 1000.0,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    early_stopping: bool = True,
    tolerance: int = 5,
) -> list[Bet]:
    """Compute the Kelly criterion using a GPU accelerated gradient-based optimizer (PyTorch).

    Args:
        list_bets (list[dict[str, Any]]): List of bets.
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

    if max_multiple > len(list_bets):
        raise ValueError(f"Error: max_multiple must not exceed {len(list_bets)}")

    # Generate combinations and bets (this part runs on CPU)
    combinations, probs = strat_utils.generate_combinations(list_bets)
    bets, book_odds = strat_utils.generate_bets_combination(list_bets, max_multiple)

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
        bet_index = [index_sel for index_sel, sel in enumerate(bet) if sel == 1]
        stake_value = stakes_final[index_bet]
        if stake_value >= 0.50:
            if len(bet_index) == 1:
                tmp_bet = list_bets[index_bet]
            else:
                tmp_bet = Bet.combine_many([list_bets[idx] for idx in bet_index])
            tmp_bet.stake = int(round(stake_value, 0))
            print(f"{tmp_bet}")
            results.append(tmp_bet)
            sum_stake += stake_value
    print(f"Bankroll used: {sum_stake:.2f} €")
    return results


def bayesian_kelly(
    list_bets: list[Bet],
    lambda_samples: dict[str, tuple[np.ndarray, np.ndarray]],
    bankroll: float = 100,
    summary: Literal["mean", "quantile"] = "quantile",
    alpha: float = 0.3,
    fraction_kelly: float = 0.5,
    per_bet_cap: float = 0.90,
):
    """Compute the optimal bet sizes using a Bayesian Kelly criterion.

    This function uses Monte Carlo samples of the scoring intensities (lambda samples)
    to estimate the posterior probabilities of match outcomes via a Skellam model.
    It then computes the Kelly fraction for each bet based on either the mean or a given quantile
    of the computed fractions. A fraction of the Kelly stake is applied along with per-bet
    capping.

    Args:
        list_bets (list[Bet]): List of bet objects for which to compute the stake.
        lambda_samples (dict[str, tuple[np.ndarray, np.ndarray]]): Mapping of match IDs to a tuple
            of numpy arrays representing the lambda samples for home and away teams.
        bankroll (float, optional): Total available bankroll. Defaults to 100.
        summary (Literal["mean", "quantile"], optional): Method to summarize the Kelly fraction
            computation. If 'mean', the average is used; if 'quantile', the quantile specified
            by alpha is used.
            Defaults to "quantile".
        alpha (float, optional): Quantile level used if summary is "quantile". Defaults to 0.3.
        fraction_kelly (float, optional): Proportion of the calculated Kelly stake to bet.
                Defaults to 0.5.
        per_bet_cap (float, optional): Maximum allowed fraction of the bankroll to wager
        on a single bet. Defaults to 0.90.

    Raises:
        ValueError: If the summary parameter is not 'mean' or 'quantile'.

    Returns:
        list[Bet]: List of bet objects with the calculated stake assigned.

    """
    mapping_match = {"H": 0, "D": 1, "A": 2}
    bets_w_stake = []
    for bet in list_bets:
        lambda_h, lambda_a = lambda_samples[bet.match_id]
        probas = strat_utils._skellam_post_probs(lh=lambda_h, la=lambda_a)
        p_samples = probas[mapping_match[bet.market]]
        X = bet.odds - 1.0  # gain unitaire si victoire
        f_kelly = (p_samples * bet.odds - (1 - p_samples)) / X  # shape (K, J)

        f_kelly = np.clip(f_kelly, 0.0, None)

        if summary == "mean":
            f_base = f_kelly.mean(axis=0)
        elif summary == "quantile":
            f_base = np.quantile(f_kelly, alpha, axis=0)
        else:
            raise ValueError("summary must be 'mean' or 'quantile'")

        f = fraction_kelly * f_base

        f = np.minimum(f, per_bet_cap)  # plafond par match

        stake = bankroll * f
        if stake > 0.0:
            tmp_bet = bet
            tmp_bet.stake = float(round(stake, 0))
            bets_w_stake.append(tmp_bet)

    sum_stake = 0.0
    possible_return = 0.0
    for bet in bets_w_stake:
        print(f"{bet}")
        sum_stake += bet.stake
        possible_return += bet.odds * bet.stake
    print(f"Bankroll used: {sum_stake:.2f} €")
    print(f"Possible return: {possible_return:.2f} €")
    return bets_w_stake


def kelly_shrinkage(
    list_bets: list[Bet],
    lambda_samples: dict[str, tuple[np.ndarray, np.ndarray]],
    per_bet_cap: float = 0.10,
    bankroll_cap: float = 0.30,
    bankroll: float = 100,
    fraction_kelly: float = 0.25,
):
    """Compute bet sizes using a shrinkage-adjusted Kelly criterion.

    This function calculates the bet stake by applying a shrinkage correction to the classic Kelly
    criterion.
    For each bet, the posterior probability is estimated using a Skellam model with lambda samples
    and its mean and variance are computed. A shrinkage factor is derived from these statistics
    and is used to adjust the full Kelly fraction. The computed fraction is then capped per bet
    and scaled by the available bankroll.
    An overall bankroll cap is also enforced if the sum of bet fractions exceeds a
    predefined threshold.

    Args:
        list_bets (list[Bet]): List of bet objects to evaluate.
        lambda_samples (dict[str, tuple[np.ndarray, np.ndarray]]):
                                                Dictionary mapping match IDs to tuples of
            numpy arrays containing lambda samples for the home and away teams.
        per_bet_cap (float, optional): Maximum fraction of the bankroll allowed on a single bet.
        Defaults to 0.10.
        bankroll_cap (float, optional): Maximum allowed total fraction of the bankroll across
        all bets. Defaults to 0.30.
        bankroll (float, optional): Total available bankroll. Defaults to 100.
        fraction_kelly (float, optional): Fraction of the shrinkage-adjusted Kelly stake to wager.
        Defaults to 0.25.

    Returns:
        list[Bet]: List of bet objects with their stake updated according to the
        shrinkage-adjusted Kelly criterion.

    """
    mapping_match = {"H": 0, "D": 1, "A": 2}
    bets_w_stake = []

    for bet in list_bets:
        lamb_h, lamb_a = lambda_samples[bet.match_id]
        probs = strat_utils._skellam_post_probs(lamb_h, lamb_a)  # shape (K,3) or (3,)
        p = probs[mapping_match[bet.market]]

        mu = p.mean(0)  # scalar for that market
        var = p.var(0, ddof=1)

        if var == 0:
            var = 1e-9

        shrink = mu * (1 - mu) / (mu * (1 - mu) + var)  # s ∈ (0,1]
        b = bet.odds - 1.0
        full = (mu * bet.odds - (1 - mu)) / b  # f⋆
        f = fraction_kelly * shrink * full  # λ-Kelly with shrink

        f = np.clip(f, 0.0, per_bet_cap)  # single-bet cap
        # bankroll-wide cap (if you really need it)
        # (collect f’s, rescale once outside the loop for clarity)

        stake = bankroll * f
        if stake > 0:
            bet.stake = int(round(stake))
            bets_w_stake.append(bet)

    total_fraction = sum(b.stake for b in bets_w_stake) / bankroll
    if total_fraction > bankroll_cap:
        scale = bankroll_cap / total_fraction
        for bet in bets_w_stake:
            bet.stake = float(round(bet.stake * scale))
    sum_stake = 0.0
    possible_return = 0.0
    for bet in bets_w_stake:
        print(f"{bet}")
        sum_stake += bet.stake
        possible_return += bet.odds * bet.stake
    print(f"Bankroll used: {sum_stake:.2f} €")
    print(f"Possible return: {possible_return:.2f} €")
    return bets_w_stake


def kelly_portfolio_torch(
    list_bets: list[Bet],
    lambda_samples: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    per_bet_cap: float = 0.10,
    bankroll_cap: float = 0.30,
    bankroll: float = 100.0,
    fraction_kelly: float = 0.25,
    iters: int = 5000,
    lr: float = 0.003,
    tol: float = 1e-8,
    verbose: bool = False,
):
    """Compute optimal bet sizes using a shrinkage-adjusted Kelly criterion with gradient-based
    optimization in PyTorch.

    This function calculates bet stakes by applying a shrinkage correction to
    the classic Kelly criterion. It uses gradient ascent to optimize the
    portfolio allocation while respecting per-bet and total bankroll constraints.

    Args:
        list_bets (list[Bet]): List of bet objects to evaluate.
        lambda_samples (dict[str, tuple[np.ndarray, np.ndarray]]): Dictionary mapping
            match IDs to tuples of numpy arrays containing lambda samples for the
            home and away teams.
        per_bet_cap (float, optional): Maximum fraction of the bankroll allowed on
            a single bet. Defaults to 0.10.
        bankroll_cap (float, optional): Maximum allowed total fraction of the bankroll
            across all bets. Defaults to 0.30.
        bankroll (float, optional): Total available bankroll. Defaults to 100.0.
        fraction_kelly (float, optional): Fraction of the shrinkage-adjusted Kelly stake
            to wager. Defaults to 0.25.
        iters (int, optional): Maximum number of iterations for gradient ascent.
            Defaults to 5000.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.003.
        tol (float, optional): Tolerance for early stopping. If the objective value
            falls below this threshold, the optimization stops. Defaults to 1e-8.
        verbose (bool, optional): If True, prints detailed information about the
            bankroll usage and possible returns. Defaults to False.

    Returns:
        list[Bet]: List of bet objects with their stake updated according to the
        shrinkage-adjusted Kelly criterion.

    Notes:
        - The shrinkage factor is derived from the mean and variance of the posterior
          probabilities, ensuring more robust stake calculations.
        - The optimization respects both per-bet and total bankroll constraints,
          projecting the solution back into the feasible region when necessary.
        - The function uses PyTorch for efficient gradient-based optimization.

    Example:
        >>> bets = [Bet(match_id="match1", market="H", odds=2.5), ...]
        >>> lambda_samples = {"match1": (np.array([...]), np.array([...]))}
        >>> optimized_bets = kelly_portfolio_torch(
                bets, lambda_samples, bankroll=100, verbose=True
            )
        >>> for bet in optimized_bets:
        >>>     print(bet)

    """

    mapping = {"H": 0, "D": 1, "A": 2}
    ps, bs = [], []

    for bet in list_bets:
        lam_h, lam_a = lambda_samples[bet.match_id]
        probs = strat_utils._skellam_post_probs(lam_h, lam_a)
        p = probs[mapping[bet.market]]

        mu, var = p.mean(), p.var(ddof=1)
        var = max(var, 1e-9)  # guard against zero variance
        shrink = mu * (1 - mu) / (mu * (1 - mu) + var)

        ps.append(mu * shrink)
        bs.append(bet.odds - 1.0)

    p_tensor = torch.as_tensor(ps, dtype=torch.float64)
    b = torch.as_tensor(bs, dtype=torch.float64)
    n = len(list_bets)

    f = torch.full((n,), bankroll_cap / n, dtype=torch.float64, requires_grad=True)

    opt = torch.optim.Adam([f], lr=lr)

    for _ in tqdm(range(iters)):
        # ----- projection to box [0, per_bet_cap] ------------------
        with torch.no_grad():
            f.clamp_(0.0, per_bet_cap)
            tot = f.sum()
            if tot > bankroll_cap:
                f.mul_(bankroll_cap / tot)

        # ----- objective (negated because we *minimise*) -----------
        opt.zero_grad()
        # log(1 + f b) and log(1 − f) are safe: caps ensure 0 < f < 1
        gain = torch.log1p(f * b)
        loss = torch.log1p(-f)
        neg_E = -(fraction_kelly * (p_tensor * gain + (1 - p_tensor) * loss).sum())

        neg_E.backward()
        opt.step()

        if neg_E.item() < tol:  # tiny objective  ⇒ done
            break

    with torch.no_grad():
        f.clamp_(0.0, per_bet_cap)
        tot = f.sum()
        if tot > bankroll_cap:
            f.mul_(bankroll_cap / tot)

    stakes = torch.round(bankroll * f).to(dtype=torch.int64).cpu().numpy()

    chosen: list[Bet] = []
    for bet, stake in zip(list_bets, stakes):
        if stake > 0.0:
            bet.stake = float(stake)
            chosen.append(bet)

    if verbose:
        used = sum(b.stake for b in chosen)
        retmax = sum(b.stake * b.odds for b in chosen)
        print(f"Bankroll used: {used:.2f} €  •  Possible return: {retmax:.2f} €")
        for b in chosen:  # type:ignore
            print(b)

    return chosen
