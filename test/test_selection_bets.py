import numpy as np
import pytest

from footix.strategy.bets import OddsInput
from footix.strategy.select_bets import select_matches_posterior


@pytest.fixture
def sample_odds_input():
    """Create sample odds input for testing."""
    return [
        OddsInput(home_team="Team1", away_team="Team2", odds=[2.0, 3.0, 4.0]),
        OddsInput(home_team="Team3", away_team="Team4", odds=[1.5, 4.0, 3.0]),
    ]


@pytest.fixture
def sample_lambda_samples():
    """Create sample lambda samples for testing."""
    # Create samples that will generate predictable probabilities
    match1_home = np.full(1000, 2.0)  # High home goals
    match1_away = np.full(1000, 1.0)  # Lower away goals

    match2_home = np.full(1000, 1.0)  # Lower home goals
    match2_away = np.full(1000, 2.0)  # Higher away goals

    return {
        "Team1 - Team2": (match1_home, match1_away),
        "Team3 - Team4": (match2_home, match2_away),
    }


def test_select_matches_single_bet():
    """Test selecting single bet per match with default thresholds."""
    odds_input = [OddsInput("Home", "Away", [2.0, 3.0, 4.0])]

    # Create samples that will favor home win
    lambda_samples = {
        "Home - Away": (np.full(1000, 2.0), np.full(1000, 0.5))  # Home lambda  # Away lambda
    }

    selected = select_matches_posterior(
        odds_input=odds_input, lambda_samples=lambda_samples, single_bet_per_game=True
    )

    assert len(selected) == 1
    assert selected[0].market == "H"
    assert selected[0].match_id == "Home - Away"
    assert selected[0].odds == 2.0


def test_select_matches_multiple_bets(sample_odds_input, sample_lambda_samples):
    """Test selecting multiple bets per match."""

    selected = select_matches_posterior(
        odds_input=sample_odds_input,
        lambda_samples=sample_lambda_samples,
        edge_floor=0.0,  # Lower threshold to get multiple bets
        prob_edge_threshold=0.5,
        single_bet_per_game=False,
    )

    assert len(selected) > 1


def test_edge_floor_filtering():
    """Test that edge floor properly filters bets."""
    odds_input = [OddsInput("Home", "Away", [1.1, 3.0, 4.0])]  # Low odds should give low edge
    lambda_samples = {"Home - Away": (np.full(1000, 2.0), np.full(1000, 0.5))}

    selected = select_matches_posterior(
        odds_input=odds_input,
        lambda_samples=lambda_samples,
        edge_floor=0.9,  # High edge requirement
    )

    assert len(selected) == 0


def test_prob_threshold_filtering():
    """Test that probability threshold properly filters bets."""
    odds_input = [OddsInput("Home", "Away", [2.0, 3.0, 4.0])]

    # Create noisy samples to get uncertain edges
    lambda_samples = {
        "Home - Away": (np.random.normal(1.5, 1.0, 1000), np.random.normal(1.5, 1.0, 1000))
    }

    selected = select_matches_posterior(
        odds_input=odds_input,
        lambda_samples=lambda_samples,
        prob_edge_threshold=0.99,  # Very high certainty requirement
    )

    assert len(selected) == 0


def test_sorting_by_edge():
    """Test that bets are sorted by descending edge."""
    odds_input = [
        OddsInput("Home1", "Away1", [2.0, 3.0, 4.0]),
        OddsInput("Home2", "Away2", [4.0, 3.0, 2.0]),
    ]

    lambda_samples = {
        "Home1 - Away1": (np.full(1000, 2.0), np.full(1000, 0.5)),
        "Home2 - Away2": (np.full(1000, 0.5), np.full(1000, 2.0)),
    }

    selected = select_matches_posterior(
        odds_input=odds_input, lambda_samples=lambda_samples, single_bet_per_game=False
    )

    # Check sorting
    for i in range(len(selected) - 1):
        assert selected[i].edge_mean >= selected[i + 1].edge_mean


def test_empty_inputs():
    """Test behavior with empty inputs."""
    selected = select_matches_posterior(
        odds_input=[],
        lambda_samples={},
    )

    assert len(selected) == 0


def test_invalid_match_id():
    """Test error handling for missing match ID in lambda samples."""
    odds_input = [OddsInput("Home", "Away", [2.0, 3.0, 4.0])]

    lambda_samples = {}  # Empty samples

    with pytest.raises(KeyError):
        select_matches_posterior(odds_input=odds_input, lambda_samples=lambda_samples)
