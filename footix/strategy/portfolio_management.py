# 0. Imports ────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from scipy.stats import norm
from torch import Tensor
from tqdm.auto import tqdm

from footix.strategy.bets import Bet


def stack_bets(bets: list[Bet]) -> tuple[np.ndarray, np.ndarray]:
    """Computes the mean and standard deviation of the edges for a list of bets.

    Args:
        bets (list[Bet]): A list of Bet objects, where each Bet contains attributes
                          `edge_mean` (float) and `edge_std` (float or None).
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                                       - The first array contains the mean edge values.
                                       - The second array contains the standard deviation
                                         of the edge values, with missing values replaced by 0.0.

    """
    mu = np.array([b.edge_mean for b in bets], dtype=float)
    sigma = np.array([b.edge_std or 0.0 for b in bets], dtype=float)
    return mu, sigma


def optimise_portfolio(
    list_bets: list[Bet],
    bankroll: float,
    max_fraction: float = 0.30,
    alpha: float = 0.05,
    gamma: float | None = None,
):
    """Optimizes the portfolio by allocating stakes to bets to maximize expected return while
    incorporating an entropy bonus and ensuring risk constraints are met.

    The function computes the optimal allocation of stakes across a list of Bet objects by
    maximizing an objective function that combines the expected value and a Shannon entropy
    term to encourage diversification. It enforces two constraints:
      1. The total stake must not exceed a specified fraction of the bankroll.
      2. The likelihood of a negative portfolio return (computed via a chance constraint) is kept
         below the significance level (alpha).

    Args:
        list_bets (list[Bet]): A list of Bet objects. Each Bet should have:
                               - edge_mean (float): Expected edge value.
                               - edge_std (float or None): Standard deviation of the edge.
        bankroll (float): The total available bankroll.
        max_fraction (float, optional): The maximum fraction of the bankroll that can be committed
                                        to stakes. Defaults to 0.30.
        alpha (float, optional): The significance level controlling the chance constraint to ensure
                                 that the probability of a negative return does not exceed alpha.
                                 Defaults to 0.05.
        gamma (float | None, optional): The entropy weight in the objective function. If None,
                                        a default value is used based on the stake cap.

    Raises:
        RuntimeError: If the optimization process fails to converge to a viable solution.

    Returns:
        list[Bet]: The list of Bet objects with updated 'stake' attributes reflecting the optimal
        allocation.

    """
    mu, sigma = stack_bets(list_bets)
    n = mu.size

    stake_cap = bankroll * max_fraction
    z_alpha = norm.ppf(1.0 - alpha)
    if gamma is None:
        gamma = 0.9 * stake_cap  # 2 % of cash-at-risk

    # ── objective with entropy bonus ───────────────────────────────────────
    def objective(stakes: np.ndarray) -> float:
        ev = np.dot(mu, stakes)
        S = np.sum(stakes) + 1e-12
        p = stakes / S
        H = -np.sum(p * np.log(p + 1e-12))
        return -(ev + gamma * H)

    # ── cash-cap linear constraint Σ s ≤ stake_cap ────────────────────────
    lin_constraint = {"type": "ineq", "fun": lambda s: stake_cap - np.sum(s)}

    # ── chance constraint P(Return < 0) ≤ α ───────────────────────────────
    def chance_fun(stakes):
        mean = np.dot(mu, stakes)
        std = np.sqrt(np.sum((sigma * stakes) ** 2))
        return mean - z_alpha * std  # ≥ 0

    chance_constraint = NonlinearConstraint(chance_fun, 0.0, np.inf)

    # ── bounds and initial guess ──────────────────────────────────────────
    bounds = Bounds(lb=np.zeros(n), ub=np.full(n, stake_cap))  # type:ignore
    x0 = np.full(n, stake_cap / n + 1e-9)

    res = minimize(
        objective,
        x0=x0,
        method="trust-constr",
        constraints=[lin_constraint, chance_constraint],
        bounds=bounds,
        options=dict(verbose=0),
    )
    if not res.success:
        raise RuntimeError(res.message)

    for b, s in zip(list_bets, res.x):
        b.stake = float(s)

    return list_bets


def optimise_portfolio_torch(
    list_bets: list[Bet],
    bankroll: float,
    max_fraction: float = 0.30,
    alpha: float = 0.05,
    gamma: float | None = None,
    lr: float = 5e-2,
    iters: int = 5_000,
    penalty_lambda: float = 1_000.0,
    verbose: bool = False,
    device: str = "cpu",
):
    """Optimizes the portfolio using a gradient-based approach implemented with PyTorch. This
    function should have the same behavior as `optimize_portfolio`. It allocates stakes to a list
    of Bet objects by optimizing an objective function that integrates the expected return and an
    entropy bonus, while imposing a chance constraint on the probability of a negative return. The
    optimization is performed using the Adam optimizer over a defined number of iterations.

    The stakes are parameterized via unconstrained raw variables that are transformed with a
    softplus function to ensure non-negativity and then scaled to satisfy the total stake
    constraint.

    Args:
        list_bets (list[Bet]): A list of Bet objects. Each Bet must have:
                               - edge_mean (float): Expected edge value.
                               - edge_std (float or None): Standard deviation of the edge.
        bankroll (float): The total available bankroll.
        max_fraction (float, optional): The maximum fraction of the bankroll that can be deployed.
                                        Defaults to 0.30.
        alpha (float, optional): The significance level for the chance constraint to ensure that
                                 the probability of a negative return is at most alpha.
                                 Defaults to 0.05.
        gamma (float | None, optional): The entropy bonus weight in the objective function.
                                        If None, a default rule-of-thumb value is applied.
        lr (float, optional): The learning rate for the Adam optimizer. Defaults to 5e-2.
        iters (int, optional): The number of iterations for the optimization loop.
                                Defaults to 5_000.
        penalty_lambda (float, optional): The penalty strength for chance constraint violations.
                                          Defaults to 1_000.0.
        verbose (bool, optional): If True, diagnostic information is printed during the
                                optimization. Defaults to False.
        device (str, optional): The device for computation (e.g., "cpu" or "cuda" for GPU).
                                Defaults to "cpu".

    Returns:
        list[Bet]: The list of Bet objects with updated 'stake' attributes based on the optimal
        allocation.

    """
    # ── 1. Static inputs ---------------------------------------------------
    mu_np, sigma_np = stack_bets(list_bets)
    mu: Tensor = torch.tensor(mu_np, device=device, dtype=torch.float)
    sigma: Tensor = torch.tensor(sigma_np, device=device, dtype=torch.float)

    n = mu.numel()
    stake_cap = bankroll * max_fraction
    z_alpha = float(norm.ppf(1.0 - alpha))
    if gamma is None:
        gamma = 0.9 * stake_cap  # ≈2 % of cash-at-risk

    # ── 2. Trainable parameters (unconstrained) ---------------------------
    #     softplus(raw) guarantees positivity; we scale later for Σs ≤ cap.
    stake_raw = torch.zeros(n, device=device, requires_grad=True, dtype=torch.float)

    # ── 3. Optimiser -------------------------------------------------------
    opt = torch.optim.Adam([stake_raw], lr=lr)

    # ── 4. Utility functions ----------------------------------------------
    def stakes_from_raw() -> Tensor:
        """Positive stakes respecting Σ s ≤ stake_cap (via scaling)"""
        s_pos = torch.nn.functional.softplus(stake_raw)  # ≥0
        S = torch.sum(s_pos) + 1e-8  # avoid /0
        scale = torch.minimum(
            torch.tensor(1.0, device=device), torch.tensor(stake_cap, device=device) / S
        ).float()
        return s_pos * scale  # Σ ≤ stake_cap

    # ── 5. Optimisation loop ----------------------------------------------
    pbar = tqdm(range(iters))
    for t in pbar:
        opt.zero_grad()
        s = stakes_from_raw()
        ev = torch.dot(mu, s)  # expected value
        # Shannon entropy of stake distribution
        p = s / (torch.sum(s) + 1e-8)
        H = -torch.sum(p * torch.log(p + 1e-8))  # nats

        # chance-constraint hinge penalty     0 if satisfied, >0 if violated
        std = torch.sqrt(torch.sum((sigma * s) ** 2))
        hinge = torch.clamp(z_alpha * std - ev, min=0.0)  # (max(⋅,0))
        penalty = penalty_lambda * hinge.pow(2)

        # final objective:  minimise –(EV + γH) + penalty
        loss = -(ev + gamma * H) + penalty
        loss.backward()
        opt.step()
        pbar.set_postfix({"loss": loss.item(), "EV": ev.item()})

    # ── 6. Write back results ---------------------------------------------
    with torch.no_grad():
        final_stakes = stakes_from_raw().cpu().numpy()

    for b, s in zip(list_bets, final_stakes):
        b.stake = float(s.round())

    # optional diagnostics
    if verbose:
        stake_sum = final_stakes.sum()
        retmax = sum((b.stake * b.odds for b in list_bets))
        print(
            f"\nTotal stake used: {stake_sum:.2f} " f"({stake_sum/bankroll*100:.1f} % of bankroll)"
        )
        print(f"\n Possible return {retmax:.1f}")
        prob_loss = float(norm.cdf(-(ev.item() / (std.item() + 1e-8))))
        print(f"P(portfolio loss)≈ {prob_loss:.3%}")
        for b in list_bets:
            print(b)

    return list_bets
