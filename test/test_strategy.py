import pytest

from footix.strategy.strategies import generate_combinations


def test_generate_combinations():
    # Cas de test : deux sélections avec probabilités
    selections = [
        {"name": "A", "probability": 0.6},
        {"name": "B", "probability": 0.3},
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
