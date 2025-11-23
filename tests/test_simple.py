import pytest

from footix.strategy.bets import Bet
from footix.strategy.simple import flat_staking


@pytest.fixture
def simple_bet():
    def _make(stake=0):
        return Bet(
            match_id="Match1",
            market="H",
            odds=1.5,
            prob_mean=0.6,
            stake=stake
        )
    return _make

@pytest.fixture
def bets_list(simple_bet):
    return [simple_bet() for _ in range(3)]

def test_flat_staking_updates_all_bets(bets_list):
    bankroll = 300
    fraction = 0.05
    updated = flat_staking(bets_list, bankroll, fraction)
    assert updated is bets_list
    for bet in updated:
        assert bet.stake == pytest.approx(15.0)

def test_flat_staking_raises_for_excessive_fraction():
    bets = [Bet(match_id=f"Match{i}", market="H", odds=1.5, prob_mean=0.6) for i in range(5)]
    bankroll = 100
    fraction = 0.3
    with pytest.raises(ValueError, match="Too many bets"):
        flat_staking(bets, bankroll, fraction)

def test_flat_staking_with_zero_bets():
    bets: list[Bet] = []
    bankroll = 100
    fraction = 0.2
    result = flat_staking(bets, bankroll, fraction)
    assert result == []
