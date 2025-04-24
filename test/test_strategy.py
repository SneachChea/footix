import pytest

from footix.strategy.bets import Bet
from footix.strategy._utils import generate_combinations


def test_generate_combinations():
    # Cas de test : deux sélections avec probabilités
    selections = [
        Bet(match_id="A", market="H", odds=42, prob_mean=0.6, edge_mean=0.5),
        Bet(match_id="B", market="A", odds=42, prob_mean=0.3, edge_mean=0.5),
    ]
    expected_combinations = [
        [0, 0],  # No event
        [1, 0],  # Only A
        [0, 1],  # Only B
        [1, 1],  # A and B
    ]
    expected_probs = [
        0.28,  # (1 - 0.6) * (1 - 0.3)
        0.42,  # 0.6 * (1 - 0.3)
        0.12,  # (1 - 0.6) * 0.3
        0.18,  # 0.6 * 0.3
    ]

    # Appeler la fonction
    combinations, probs = generate_combinations(selections)

    # Vérifier que les combinaisons sont correctes
    assert (
        combinations == expected_combinations
    ), f"Expected {expected_combinations}, but got {combinations}"

    # Vérifier que les probabilités sont correctes
    assert (
        pytest.approx(probs, rel=1e-6) == expected_probs
    ), f"Expected {expected_probs}, but got {probs}"
