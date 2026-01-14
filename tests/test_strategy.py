import pytest

from footix.strategy._utils import generate_combinations
from footix.strategy.bets import Bet


def test_generate_combinations():
    """Test the generate_combinations function with various scenarios."""
    # Test case 1: Two selections with probabilities
    selections_1 = [
        Bet(match_id="A", market="H", odds=2.0, prob_mean=0.6),
        Bet(match_id="B", market="A", odds=3.0, prob_mean=0.3),
    ]
    expected_combinations_1 = [
        [0, 0],  # No event
        [1, 0],  # Only A
        [0, 1],  # Only B
        [1, 1],  # A and B
    ]
    expected_probs_1 = [
        0.28,  # (1 - 0.6) * (1 - 0.3)
        0.42,  # 0.6 * (1 - 0.3)
        0.12,  # (1 - 0.6) * 0.3
        0.18,  # 0.6 * 0.3
    ]

    combinations_1, probs_1 = generate_combinations(selections_1)
    assert combinations_1 == expected_combinations_1
    assert pytest.approx(probs_1, rel=1e-6) == expected_probs_1
    assert pytest.approx(sum(probs_1)) == 1.0  # Probabilities should sum to 1

    # Test case 2: Single selection
    selections_2 = [Bet(match_id="C", market="D", odds=3.5, prob_mean=0.4)]
    expected_combinations_2 = [
        [0],  # No event
        [1],  # Event occurs
    ]
    expected_probs_2 = [
        0.6,  # 1 - 0.4
        0.4,  # 0.4
    ]

    combinations_2, probs_2 = generate_combinations(selections_2)
    assert combinations_2 == expected_combinations_2
    assert pytest.approx(probs_2, rel=1e-6) == expected_probs_2

    # Test case 3: Empty selections list
    selections_3 = []
    combinations_3, probs_3 = generate_combinations(selections_3)
    assert combinations_3 == [[]]
    assert probs_3 == [1.0]

    # Test case 4: Three selections
    selections_4 = [
        Bet(match_id="D", market="H", odds=2.0, prob_mean=0.5),
        Bet(match_id="E", market="A", odds=2.0, prob_mean=0.5),
        Bet(match_id="F", market="D", odds=2.0, prob_mean=0.5),
    ]
    combinations_4, probs_4 = generate_combinations(selections_4)
    assert len(combinations_4) == 2**3  # Should have 8 combinations
    assert pytest.approx(sum(probs_4)) == 1.0  # Probabilities should sum to 1
    assert all(
        len(comb) == 3 for comb in combinations_4
    )  # Each combination should have 3 elements
