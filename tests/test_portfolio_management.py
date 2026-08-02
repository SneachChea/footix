from footix.strategy.bets import Bet
from footix.strategy.kelly_strategies import fractional_kelly
from footix.strategy.portfolio_management import optimise_portfolio_torch


def test_torch_portfolio_handles_empty_input() -> None:
    assert optimise_portfolio_torch([], bankroll=100.0) == []


def test_torch_portfolio_respects_rounded_bankroll_cap() -> None:
    bets = [
        Bet("m1", "H", 2.2, 0.55, edge_std=0.1),
        Bet("m2", "H", 2.1, 0.54, edge_std=0.1),
    ]
    result = optimise_portfolio_torch(
        bets,
        bankroll=100.0,
        max_fraction=0.30,
        gamma=0.0,
        iters=50,
        verbose=False,
    )

    assert sum(bet.stake for bet in result) <= 30.0


def test_fractional_kelly_respects_total_cap() -> None:
    bets = [Bet("m1", "H", 2.0, 0.60), Bet("m2", "H", 2.0, 0.60)]
    result = fractional_kelly(bets, bankroll=100.0, fraction_kelly=0.25, max_fraction=0.30)

    assert sum(bet.stake for bet in result) <= 30.0
    assert all(bet.stake > 0 for bet in result)
