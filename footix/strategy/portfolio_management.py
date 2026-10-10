from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from scipy.stats import norm
from torch import Tensor
from tqdm.auto import tqdm

from footix.strategy.bets import Bet


@dataclass(frozen=True)
class PortfolioScenarios:
    """Joint P&L scenarios for a set of bets.

    Attributes:
        returns (np.ndarray): Array of shape ``(n_scenarios, n_bets)`` giving
            the return per euro staked (``odds_i * won - 1``) of each bet in
            each scenario.
        bet_keys (tuple[tuple[str, str], ...]): ``(match_id, market)`` keys in
            column order, used to verify the matrix is aligned with the bets
            being optimised.

    """

    returns: np.ndarray
    bet_keys: tuple[tuple[str, str], ...]


def stack_bets(bets: list[Bet]) -> tuple[np.ndarray, np.ndarray]:
    """Computes the mean and standard deviation of the edges for a list of bets.

    Args:
        bets (list[Bet]): A list of Bet objects, where each Bet contains attributes
                          `edge_mean` (float) and `edge_std` (float or None).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                                       - The first array contains the mean edge values.
                                          - The second array contains risk values.

                                        Missing values use Bernoulli P&L risk.

    """
    mu = np.array([b.edge_mean for b in bets], dtype=float)
    sigma = np.array(
        [
            b.edge_std
            if b.edge_std is not None
            else b.odds * np.sqrt(b.prob_mean * (1.0 - b.prob_mean))
            for b in bets
        ],
        dtype=float,
    )
    return mu, sigma


def stack_markowitz(bets: list[Bet]) -> tuple[np.ndarray, np.ndarray]:
    """Computes the mean and standard deviation of the bet (and not from the model)

    Args:
        bets (list[Bet]): A list of Bet objects, where each Bet contains attributes

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                                        - The first array contains the mean edge values.
                                        - The second array contains the standard deviation
    """
    mu = np.array([b.edge_mean for b in bets], dtype=float)
    sigma = np.array([b.odds**2 * b.prob_mean * (1.0 - b.prob_mean) for b in bets], dtype=float)
    return mu, sigma


def optimise_portfolio(
    list_bets: list[Bet],
    bankroll: float,
    max_fraction: float = 0.30,
    alpha: float = 0.05,
    gamma: float | None = None,
):
    """Optimizes bet stakes using SciPy's constrained optimization to maximize return with risk
    control.

    This function uses classical constrained optimization to find the optimal stake allocation
    that maximizes expected value while maintaining risk constraints. It incorporates Shannon
    entropy to encourage diversification.

    Args:
        list_bets (list[Bet]): List of bets to optimize. Each Bet must have edge_mean and edge_std.
        bankroll (float): Total available funds for betting.
        max_fraction (float, optional): Maximum fraction of bankroll to stake. Defaults to 0.30.
        alpha (float, optional): Risk threshold for chance constraint (probability of loss).
            Defaults to 0.05.
        gamma (float | None, optional): Entropy bonus weight.
            If None, defaults to 0.9 * stake_cap.
            Controls diversification strength.

    Returns:
        list[Bet]: Input bets with optimized stakes set in their stake attribute.

    Raises:
        RuntimeError: If the optimization fails to converge to a valid solution.

    Notes:
        - Uses trust-region constrained optimization from SciPy
        - Enforces two main constraints:
            1. Total stakes ≤ max_fraction * bankroll
            2. P(portfolio loss) ≤ alpha via chance constraint

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
    scenarios: PortfolioScenarios,
    max_fraction: float = 0.30,
    lr: float = 5e-2,
    iters: int = 5_000,
    verbose: bool = False,
    device: str = "cpu",
):
    """Optimizes bet stakes using PyTorch gradient descent on scenario Kelly.

    Maximizes the expected log-growth of the bankroll over joint P&L
    scenarios:

        G(f) = (1/M) * sum_m log(1 + R_m @ f)

    where ``R_m`` is the vector of per-euro returns of the bets in scenario
    ``m`` and ``f`` the fractions of bankroll staked. Scenario returns keep
    the discrete, asymmetric payoff structure (``odds - 1`` or ``-1``) and
    the joint dependence across bets, so no normality assumption is needed.

    The total exposure is parameterized as ``max_fraction * sigmoid(a)`` so
    the optimiser can choose to keep cash, and the per-bet weights are a
    softmax so each stake is non-negative.

    Args:
        list_bets (list[Bet]): List of bets to optimize.
        bankroll (float): Total available funds for betting.
        scenarios (PortfolioScenarios): Joint P&L scenarios, one column per
            bet in the same order as ``list_bets``.
        max_fraction (float, optional): Maximum fraction of bankroll to stake.
            Must be strictly below 1 so the simulated bankroll stays positive.
            Defaults to 0.30.
        lr (float, optional): Learning rate for Adam optimizer. Defaults to 5e-2.
        iters (int, optional): Number of optimization iterations. Defaults to 5_000.
        verbose (bool, optional): Whether to print diagnostic information.
            Defaults to False.
        device (str, optional): PyTorch device to use ('cpu' or 'cuda').
            Defaults to "cpu".

    Returns:
        list[Bet]: Input bets with optimized stakes set in their stake attribute.

    Raises:
        ValueError: If inputs are invalid or the scenarios are not aligned
            with the bets.
        RuntimeError: If the optimization produces a non-finite loss.

    """
    if not list_bets:
        return list_bets
    if bankroll <= 0:
        raise ValueError("bankroll must be positive")
    if not 0 < max_fraction < 1:
        raise ValueError("max_fraction must be in (0, 1)")
    if lr <= 0 or iters <= 0:
        raise ValueError("lr and iters must be positive")

    if scenarios.returns.ndim != 2:
        raise ValueError("scenario returns must be a 2D array")
    n_scenarios, n_bets = scenarios.returns.shape
    if n_scenarios == 0:
        raise ValueError("scenarios must contain at least one row")
    if n_bets != len(list_bets):
        raise ValueError("scenarios must have one column per bet")
    expected_keys = tuple((bet.match_id, bet.market) for bet in list_bets)
    if len(scenarios.bet_keys) != n_bets or scenarios.bet_keys != expected_keys:
        raise ValueError("scenarios.bet_keys do not match the bets passed in")
    if not np.all(np.isfinite(scenarios.returns)):
        raise ValueError("scenario returns must be finite")

    returns_t: Tensor = torch.tensor(scenarios.returns, device=device, dtype=torch.float64)
    stake_cap = bankroll * max_fraction

    exposure_raw = torch.zeros(1, device=device, requires_grad=True, dtype=torch.float64)
    weight_raw = torch.zeros(n_bets, device=device, requires_grad=True, dtype=torch.float64)
    opt = torch.optim.Adam([exposure_raw, weight_raw], lr=lr)

    def fractions_from_raw() -> Tensor:
        """Non-negative fractions whose sum stays strictly below max_fraction."""
        exposure = max_fraction * torch.sigmoid(exposure_raw)
        weights = torch.softmax(weight_raw, dim=0)
        return exposure * weights

    pbar = tqdm(range(iters), disable=not verbose)
    for _ in pbar:
        opt.zero_grad()
        fractions = fractions_from_raw()
        portfolio_returns = returns_t @ fractions
        if torch.min(1.0 + portfolio_returns) <= 0.0:
            raise RuntimeError("Portfolio returns leave the bankroll non-positive")
        log_growth = torch.mean(torch.log1p(portfolio_returns))
        loss = -log_growth
        if not torch.isfinite(loss):
            raise RuntimeError("Portfolio optimization produced a non-finite loss")
        loss.backward()
        opt.step()

    with torch.no_grad():
        final_fractions = fractions_from_raw().cpu().numpy()

    final_stakes = final_fractions * bankroll
    rounded = np.rint(final_stakes).astype(int)
    rounded = np.maximum(rounded, 0)
    cap = int(np.floor(stake_cap))
    excess = max(int(rounded.sum()) - cap, 0)
    for index in np.argsort(-rounded):
        if excess == 0:
            break
        reduction = min(excess, int(rounded[index]))
        rounded[index] -= reduction
        excess -= reduction
    for b, s in zip(list_bets, rounded):
        b.stake = float(s)

    if verbose:
        # Diagnostics on the rounded stakes actually returned.
        stake_sum = rounded.sum()
        stake_fractions = torch.tensor(rounded, dtype=torch.float64) / bankroll
        with torch.no_grad():
            portfolio_returns = (returns_t @ stake_fractions).numpy()
        log_returns = np.log1p(portfolio_returns)
        g_hat = float(np.mean(log_returns))
        se_g = (
            float(np.std(log_returns, ddof=1) / np.sqrt(n_scenarios)) if n_scenarios > 1 else 0.0
        )
        tail_q = float(np.quantile(portfolio_returns, 0.05))
        tail_return = float(np.mean(portfolio_returns[portfolio_returns <= tail_q]))
        print(f"\nTotal stake used: {stake_sum:.2f} ({stake_sum/bankroll*100:.1f} % of bankroll)")
        print(f"Mean log-growth G: {g_hat:.6f} (SE {se_g:.6f})")
        print(f"Mean return: {float(portfolio_returns.mean()):.2%}")
        print(f"P(loss): {(portfolio_returns < 0).mean():.2%}")
        print(f"Q5% return: {tail_q:.2%} | Tail return (5% worst): {tail_return:.2%}")
        print(f"Worst scenario: {float(portfolio_returns.min()):.2%}")
        for b in list_bets:
            print(b)

    return list_bets


def bayesian_portfolio_optim(
    list_bets: list[Bet],
    edge_samples: Mapping[Any, Any],
    bankroll: float,
    max_fraction: float = 0.30,
    gamma: float | None = None,
    lr: float = 5e-2,
    iters: int = 3_000,
    verbose: bool = False,
    device: str = "cpu",
) -> list[Bet]:
    """Optimizes bet stakes using Bayesian posterior samples of edge.

    This function implements portfolio optimization by maximizing the expected
    gain marginalized over the uncertainty in edge estimates. It uses Monte Carlo
    samples from the posterior distribution of edge for each bet.

    The optimization maximizes:
        E_θ[Gain] = (1/K) * Σ_k Σ_i edge_i^(k) * s_i

    where edge_i^(k) is the k-th posterior sample of edge for bet i.

    Args:
        list_bets: List of bets to optimize. Each Bet must have a match_id and market.
        edge_samples: Dictionary mapping (match_id, market) tuple or match_id to
            numpy arrays of posterior edge samples. Shape: (n_samples,) for each bet.
        bankroll: Total available funds for betting.
        max_fraction: Maximum fraction of bankroll to stake. Defaults to 0.30.
        gamma: Entropy bonus weight for diversification. If None, defaults to
            0.5 * stake_cap.
        lr: Learning rate for Adam optimizer. Defaults to 5e-2.
        iters: Number of optimization iterations. Defaults to 3_000.
        verbose: Whether to print diagnostic information. Defaults to False.
        device: PyTorch device to use ('cpu' or 'cuda'). Defaults to "cpu".

    Returns:
        list[Bet]: Input bets with optimized stakes set in their stake attribute.

    Notes:
        - Uses Adam optimizer with gradient descent
        - Enforces constraints softly through scaling:
            1. Stakes positivity via softplus
            2. Total stakes via scaling to respect stake_cap
        - Includes Shannon entropy term for diversification
        - Marginalizes over posterior uncertainty in edge estimates

    Example:
        >>> # edge_samples[bet.match_id] contains posterior samples for each outcome
        >>> edge_samples = {
        ...     ("match1", "H"): np.random.normal(0.1, 0.05, 1000),
        ...     ("match1", "D"): np.random.normal(-0.05, 0.03, 1000),
        ... }
        >>> optimized_bets = bayesian_portfolio_optim(bets, edge_samples, bankroll=1000)

    """
    n = len(list_bets)
    stake_cap = bankroll * max_fraction

    if gamma is None:
        gamma = 0.5 * stake_cap

    # ── 1. Build edge samples matrix (n_bets, n_samples) ──────────────────
    edge_matrix_list = []
    for bet in list_bets:
        # Support both (match_id, market) tuple keys and match_id keys
        key = (bet.match_id, bet.market)
        if key in edge_samples:
            samples = edge_samples[key]
        elif bet.match_id in edge_samples:
            # Fallback: assume edge_samples[match_id] is already the right samples
            samples = edge_samples[bet.match_id]
        else:
            raise KeyError(f"No edge samples found for bet {bet.match_id}, market {bet.market}")
        edge_matrix_list.append(samples)

    # Convert to tensor: shape (n_bets, n_samples)
    edge_matrix_np = np.stack(edge_matrix_list, axis=0)
    edge_matrix: Tensor = torch.tensor(edge_matrix_np, device=device, dtype=torch.float)

    # ── 2. Initialize stakes and optimizer ────────────────────────────────
    stake_raw = torch.zeros(n, device=device, requires_grad=True, dtype=torch.float)
    opt = torch.optim.Adam([stake_raw], lr=lr)

    def stakes_from_raw() -> Tensor:
        """Positive stakes respecting Σ s ≤ stake_cap (via scaling)."""
        s_pos = torch.nn.functional.softplus(stake_raw)
        total = torch.sum(s_pos) + 1e-8
        scale = torch.minimum(
            torch.tensor(1.0, device=device),
            torch.tensor(stake_cap, device=device) / total,
        ).float()
        return s_pos * scale

    # ── 3. Optimization loop ──────────────────────────────────────────────
    pbar = tqdm(range(iters))
    for _ in pbar:
        opt.zero_grad()
        s = stakes_from_raw()

        # Compute expected gain marginalized over samples
        # gain_per_sample[k] = Σ_i edge_i^(k) * s_i
        # E[gain] = (1/K) * Σ_k gain_per_sample[k]
        gain_per_sample = torch.matmul(s, edge_matrix)  # shape: (n_samples,)
        expected_gain = torch.mean(gain_per_sample)

        # Entropy bonus for diversification
        p = s / (torch.sum(s) + 1e-8)
        entropy = -torch.sum(p * torch.log(p + 1e-8))

        # Compute variance of gain across samples (for monitoring)
        gain_std = torch.std(gain_per_sample)

        # Loss: negative of (expected gain + entropy bonus)
        loss = -(expected_gain + gamma * entropy)

        loss.backward()
        opt.step()
        pbar.set_postfix(
            {
                "loss": loss.item(),
                "E[gain]": expected_gain.item(),
                "std[gain]": gain_std.item(),
            }
        )

    # ── 4. Extract final stakes ───────────────────────────────────────────
    with torch.no_grad():
        final_stakes = stakes_from_raw().cpu().numpy()

    for bet, stake in zip(list_bets, final_stakes):
        bet.stake = float(stake.round())

    if verbose:
        stake_sum = final_stakes.sum()
        print(
            f"\nTotal stake used: {stake_sum:.2f} "
            f"({stake_sum / bankroll * 100:.1f} % of bankroll)"
        )
        # Compute posterior statistics
        with torch.no_grad():
            s_tensor = torch.tensor(final_stakes, device=device, dtype=torch.float)
            gains = torch.matmul(s_tensor, edge_matrix).cpu().numpy()
        print(f"E[Gain]: {gains.mean():.2f} ± {gains.std():.2f}")
        print(f"P(Gain > 0): {(gains > 0).mean():.1%}")
        print(f"5th percentile: {np.percentile(gains, 5):.2f}")
        print(f"95th percentile: {np.percentile(gains, 95):.2f}")
        for bet in list_bets:
            print(bet)

    return list_bets


def mean_variance_markowitz(
    list_bets: list[Bet],
    bankroll: float,
    risk_aversion: float = 1.0,
    lr: float = 1e-2,
    iters: int = 5000,
    device: str = "cpu",
    verbose: bool = False,
):
    mu_np, sigma_np = stack_markowitz(list_bets)
    mu: Tensor = torch.tensor(mu_np, device=device, dtype=torch.float)
    sigma: Tensor = torch.tensor(sigma_np, device=device, dtype=torch.float)

    stake_raw = torch.zeros(mu.numel(), device=device, requires_grad=True, dtype=torch.float)
    optimizer = torch.optim.Adam([stake_raw], lr=lr)

    def get_valid_stakes() -> Tensor:
        s_pos = torch.nn.functional.softplus(stake_raw)
        S = torch.sum(s_pos) + 1e-8  # avoid /0
        scale = torch.minimum(
            torch.tensor(1.0, device=device), torch.tensor(bankroll, device=device) / S
        ).float()
        return s_pos * scale

    pbar = tqdm(range(iters))
    for t in pbar:
        optimizer.zero_grad()
        s = get_valid_stakes()
        ev = torch.dot(mu, s)
        var = torch.sum((sigma * s**2))
        if verbose:
            print("var", var)
            print("ev", ev)

        loss = -(ev - risk_aversion * var)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item(), "EV": ev.item()})

    with torch.no_grad():
        final_stakes = get_valid_stakes().cpu().numpy()
    for b, s in zip(list_bets, final_stakes):
        b.stake = float(s.round())
    if verbose:
        stake_sum = final_stakes.sum()
        retmax = sum((b.stake * b.odds for b in list_bets))
        print(
            f"\nTotal stake used: {stake_sum:.2f} " f"({stake_sum/bankroll*100:.1f} % of bankroll)"
        )
        print(f"\n Possible return {retmax:.1f}")
        for b in list_bets:
            print(b)
    return list_bets
