# 0. Imports ────────────────────────────────────────────────────────────────
from __future__ import annotations
from footix.strategy.bets import Bet
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from torch import Tensor
import torch
from tqdm.auto import tqdm

# 2. Helper to build numpy arrays from a list[Bet] ──────────────────────────
def stack_bets(bets: list[Bet]) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
        mu  : shape (n,)   vector of edge means
        sigma: shape (n,)   vector of edge std-devs (0 where unknown)
    """
    mu    = np.array([b.edge_mean             for b in bets],  dtype=float)
    sigma = np.array([b.edge_std or 0.0       for b in bets],  dtype=float)
    return mu, sigma

def optimise_portfolio(
    bets: list[Bet],
    bankroll: float,
    max_fraction: float = 0.30,
    alpha: float = 0.05,
    gamma: float | None = None,            # <-- new; if None choose rule-of-thumb
):
    mu, sigma = stack_bets(bets)
    n         = mu.size

    stake_cap = bankroll * max_fraction
    z_alpha   = norm.ppf(1.0 - alpha)
    if gamma is None:
        gamma = 0.9*stake_cap          # 2 % of cash-at-risk

    # ── objective with entropy bonus ───────────────────────────────────────
    def objective(stakes: np.ndarray) -> float:
        ev  = np.dot(mu, stakes)
        S   = np.sum(stakes) + 1e-12
        p   = stakes / S
        H   = -np.sum(p * np.log(p + 1e-12))
        return -(ev + gamma * H)

    # ── cash-cap linear constraint Σ s ≤ stake_cap ────────────────────────
    lin_constraint = {"type": "ineq",
                      "fun": lambda s: stake_cap - np.sum(s)}

    # ── chance constraint P(Return < 0) ≤ α ───────────────────────────────
    def chance_fun(stakes):
        mean = np.dot(mu, stakes)
        std  = np.sqrt(np.sum((sigma * stakes) ** 2))
        return mean - z_alpha * std        # ≥ 0

    chance_constraint = NonlinearConstraint(chance_fun, 0., np.inf)

    # ── bounds and initial guess ──────────────────────────────────────────
    bounds = Bounds(lb=np.zeros(n), ub=np.full(n, stake_cap))
    x0     = np.full(n, stake_cap / n + 1e-9)

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

    for b, s in zip(bets, res.x):
        b.stake = float(s)

    return bets

def optimise_portfolio_torch(
    bets: list[Bet],
    bankroll      : float,
    max_fraction  : float = 0.30,      # bankroll you are *willing* to deploy
    alpha         : float = 0.05,      # P(total profit < 0) ≤ α
    gamma         : float | None = None,  # entropy weight (None → rule-of-thumb)
    lr            : float = 5e-2,      # Adam learning rate
    iters         : int   = 5_000,     # optimisation steps
    penalty_lambda: float = 1_000.0,   # strength of chance-constraint penalty
    verbose       : bool  = False,
    device        : str   = "cpu",     # put "cuda" if you have a GPU
):
    """
    Fills Bet.stake in place and returns the list.
    All tensors are float32 for speed.
    """
    # ── 1. Static inputs ---------------------------------------------------
    mu_np, sigma_np = stack_bets(bets)
    mu    : Tensor = torch.tensor(mu_np,    device=device, dtype=torch.float)
    sigma : Tensor = torch.tensor(sigma_np, device=device, dtype=torch.float)

    n           = mu.numel()
    stake_cap   = bankroll * max_fraction
    z_alpha     = float(norm.ppf(1.0 - alpha))
    if gamma is None:
        gamma = 0.9*stake_cap          # ≈2 % of cash-at-risk

    # ── 2. Trainable parameters (unconstrained) ---------------------------
    #     softplus(raw) guarantees positivity; we scale later for Σs ≤ cap.
    stake_raw = torch.zeros(n, device=device, requires_grad=True, dtype=torch.float)

    # ── 3. Optimiser -------------------------------------------------------
    opt = torch.optim.Adam([stake_raw], lr=lr)

    # ── 4. Utility functions ----------------------------------------------
    def stakes_from_raw() -> Tensor:
        """positive stakes respecting Σ s ≤ stake_cap (via scaling)"""
        s_pos  = torch.nn.functional.softplus(stake_raw)      # ≥0
        S      = torch.sum(s_pos) + 1e-8                      # avoid /0
        scale  = torch.minimum(torch.tensor(1.0, device=device),
                               torch.tensor(stake_cap, device=device) / S).float()
        return s_pos * scale                                  # Σ ≤ stake_cap

    # ── 5. Optimisation loop ----------------------------------------------
    pbar = tqdm(range(iters))
    for t in pbar:
        opt.zero_grad()
        s   = stakes_from_raw()
        ev  = torch.dot(mu, s)                                # expected value
        # Shannon entropy of stake distribution
        p   = s / (torch.sum(s) + 1e-8)
        H   = -torch.sum(p * torch.log(p + 1e-8))             # nats

        # chance-constraint hinge penalty     0 if satisfied, >0 if violated
        std = torch.sqrt(torch.sum((sigma * s) ** 2))
        hinge = torch.clamp(z_alpha * std - ev, min=0.0)      # (max(⋅,0))
        penalty = penalty_lambda * hinge.pow(2)

        # final objective:  minimise –(EV + γH) + penalty
        loss = -(ev + gamma * H) + penalty
        loss.backward()
        opt.step()
        pbar.set_postfix({"loss": loss.item(), "EV": ev.item()})

    # ── 6. Write back results ---------------------------------------------
    with torch.no_grad():
        final_stakes = stakes_from_raw().cpu().numpy()

    for b, s in zip(bets, final_stakes):
        b.stake = float(round(s, 0))

    # optional diagnostics
    if verbose:
        stake_sum = final_stakes.sum()
        print(f"\nTotal stake used: {stake_sum:.2f} "
              f"({stake_sum/bankroll*100:.1f} % of bankroll)")
        prob_loss = float(norm.cdf(- (ev.item()/ (std.item()+1e-8))))
        print(f"P(portfolio loss)≈ {prob_loss:.3%}")
        for b in bets:
            print(b)

    return bets