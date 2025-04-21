import pytest

from footix.strategy.bets import Bet


def test_bet_creation():
    bet = Bet("match1", "H", 2.0, 0.10, 0.55)
    assert bet.match_id == "match1"
    assert bet.market == "H"
    assert bet.odds == 2.0
    assert bet.edge_mean == 0.10
    assert bet.prob_mean == 0.55
    assert bet.edge_std is None
    assert bet.prob_edge_pos is None
    assert bet.stake is None


def test_to_dict():
    bet = Bet("match1", "D", 3.5, 0.20, 0.40, edge_std=0.05, stake=100)
    d = bet.to_dict()
    assert d["match_id"] == "match1"
    assert d["market"] == "D"
    assert d["odds"] == 3.5
    assert d["stake"] == 100


def test_str_and_repr():
    bet = Bet("m1", "A", 2.0, 0.12, 0.60, stake=50)
    s = str(bet)
    r = repr(bet)
    assert "[m1 | A]" in s
    assert "Bet(" in r
    assert "odds=2.0" in r


def test_add_operator():
    b1 = Bet("m1", "H", 2.0, 0.1, 0.5)
    b2 = Bet("m2", "A", 1.5, 0.08, 0.6)
    combined = b1 + b2
    assert combined.match_id == "m1 + m2"
    assert combined.market == "H + A"
    assert abs(combined.odds - 3.0) < 1e-6
    assert abs(combined.prob_mean - 0.3) < 1e-6
    assert combined.edge_mean is not None


def test_iadd_operator():
    b1 = Bet("m1", "H", 2.0, 0.1, 0.5)
    b2 = Bet("m2", "D", 2.0, 0.2, 0.4)
    b1 += b2
    assert b1.match_id == "m1 + m2"
    assert b1.market == "H + D"
    assert abs(b1.odds - 4.0) < 1e-6
    assert abs(b1.prob_mean - 0.2) < 1e-6


def test_combine_many():
    bets = [
        Bet("m1", "H", 2.0, 0.1, 0.5),
        Bet("m2", "D", 2.0, 0.1, 0.5),
        Bet("m3", "A", 2.0, 0.1, 0.5),
    ]
    combined = Bet.combine_many(bets)
    assert combined.match_id == "m1 + m2 + m3"
    assert combined.market == "H + D + A"
    assert abs(combined.odds - 8.0) < 1e-6
    assert abs(combined.prob_mean - 0.125) < 1e-6


def test_combine_empty_raises():
    with pytest.raises(ValueError):
        Bet.combine_many([])
