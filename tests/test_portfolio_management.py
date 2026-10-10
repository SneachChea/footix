import numpy as np
import pytest

from footix.strategy.bets import Bet
from footix.strategy.kelly_strategies import fractional_kelly
from footix.strategy.portfolio_management import PortfolioScenarios, optimise_portfolio_torch


def _scenarios(
    bets: list[Bet], win_prob: list[float], rng: np.random.Generator
) -> PortfolioScenarios:
    """Independent Bernoulli scenarios matching each bet's odds/probability."""
    n = 10_000
    won = rng.random((n, len(bets))) < np.asarray(win_prob)
    odds = np.array([bet.odds for bet in bets], dtype=float)
    returns = odds[None, :] * won - 1.0
    keys = tuple((bet.match_id, bet.market) for bet in bets)
    return PortfolioScenarios(returns=returns, bet_keys=keys)


def test_torch_portfolio_handles_empty_input() -> None:
    scenarios = PortfolioScenarios(np.empty((0, 0)), ())
    assert optimise_portfolio_torch([], bankroll=100.0, scenarios=scenarios) == []


def test_torch_portfolio_respects_rounded_bankroll_cap() -> None:
    bets = [Bet("m1", "H", 2.2, 0.55), Bet("m2", "H", 2.1, 0.54)]
    scenarios = _scenarios(bets, [0.55, 0.54], np.random.default_rng(0))
    result = optimise_portfolio_torch(
        bets,
        bankroll=100.0,
        scenarios=scenarios,
        max_fraction=0.30,
        iters=200,
        verbose=False,
    )

    assert sum(bet.stake for bet in result) <= 30.0


def test_torch_portfolio_rejects_misaligned_scenarios() -> None:
    bets = [Bet("m1", "H", 2.2, 0.55), Bet("m2", "H", 2.1, 0.54)]
    scenarios = _scenarios(bets, [0.55, 0.54], np.random.default_rng(1))
    scenarios = PortfolioScenarios(scenarios.returns, (("m1", "H"), ("m3", "A")))

    with pytest.raises(ValueError, match="bet_keys"):
        optimise_portfolio_torch(bets, bankroll=100.0, scenarios=scenarios)


def test_torch_portfolio_rejects_non_finite_returns() -> None:
    bets = [Bet("m1", "H", 2.2, 0.55)]
    scenarios = PortfolioScenarios(np.full((100, 1), np.nan), (("m1", "H"),))

    with pytest.raises(ValueError, match="finite"):
        optimise_portfolio_torch(bets, bankroll=100.0, scenarios=scenarios)


def test_torch_portfolio_rejects_max_fraction_at_one() -> None:
    bets = [Bet("m1", "H", 2.2, 0.55)]
    scenarios = _scenarios(bets, [0.55], np.random.default_rng(2))

    with pytest.raises(ValueError, match="max_fraction"):
        optimise_portfolio_torch(bets, bankroll=100.0, scenarios=scenarios, max_fraction=1.0)


def test_kelly_tilts_toward_lower_variance_bet() -> None:
    # Same mean edge (~+0.5), very different variance.
    high_var = Bet("m1", "H", 3.0, 0.5)
    low_var = Bet("m2", "H", 1.6667, 0.9)
    scenarios = _scenarios([high_var, low_var], [0.5, 0.9], np.random.default_rng(3))

    result = optimise_portfolio_torch(
        [high_var, low_var],
        bankroll=100.0,
        scenarios=scenarios,
        max_fraction=0.30,
        iters=500,
        verbose=False,
    )

    assert result[1].stake > result[0].stake


def test_torch_portfolio_deterministic_given_scenarios() -> None:
    bets = [Bet("m1", "H", 2.2, 0.55), Bet("m2", "H", 2.1, 0.54)]
    scenarios = _scenarios(bets, [0.55, 0.54], np.random.default_rng(5))

    def run() -> list[float]:
        fresh = [Bet("m1", "H", 2.2, 0.55), Bet("m2", "H", 2.1, 0.54)]
        optimise_portfolio_torch(fresh, bankroll=100.0, scenarios=scenarios, iters=100)
        return [bet.stake for bet in fresh]

    assert run() == run()


def test_fractional_kelly_respects_total_cap() -> None:
    bets = [Bet("m1", "H", 2.0, 0.60), Bet("m2", "H", 2.0, 0.60)]
    result = fractional_kelly(bets, bankroll=100.0, fraction_kelly=0.25, max_fraction=0.30)

    assert sum(bet.stake for bet in result) <= 30.0
    assert all(bet.stake > 0 for bet in result)
