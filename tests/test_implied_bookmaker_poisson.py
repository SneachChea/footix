import numpy as np
import pytest

from footix.models.utils import implied_poisson_goals
from footix.utils.typing import ProbaResult


@pytest.fixture
def sample_bookmaker_proba():
    """Fixture providing sample bookmaker probabilities."""
    return ProbaResult(
        proba_home=0.45,  # Home win probability
        proba_draw=0.25,  # Draw probability
        proba_away=0.30,  # Away win probability
    )


def test_implied_poisson_goals_basic(sample_bookmaker_proba):
    """Test basic functionality of implied_poisson_goals."""
    result = implied_poisson_goals(sample_bookmaker_proba)

    # Check return type
    assert hasattr(result, "home_goals_probs")
    assert hasattr(result, "away_goals_probs")

    # Check probabilities sum to approximately 1
    assert np.isclose(np.sum(result.home_goals_probs), 1.0, atol=1e-3)
    assert np.isclose(np.sum(result.away_goals_probs), 1.0, atol=1e-3)

    # Check all probabilities are positive
    assert np.all(result.home_goals_probs >= 0)
    assert np.all(result.away_goals_probs >= 0)


def test_implied_poisson_goals_custom_parameters(sample_bookmaker_proba):
    """Test with custom k_sum and nbr_goals parameters."""
    result = implied_poisson_goals(sample_bookmaker_proba, k_sum=30, nbr_goals=5)

    # Check length of probability arrays
    assert len(result.home_goals_probs) == 5
    assert len(result.away_goals_probs) == 5


def test_implied_poisson_goals_extreme_probabilities():
    """Test with extreme probability values."""
    extreme_proba = ProbaResult(
        proba_home=0.9,  # Very high home win probability
        proba_draw=0.05,  # Very low draw probability
        proba_away=0.05,  # Very low away win probability
    )

    result = implied_poisson_goals(extreme_proba)

    # Home team should have higher expected goals
    expected_home_goals = np.sum(np.arange(len(result.home_goals_probs)) * result.home_goals_probs)
    expected_away_goals = np.sum(np.arange(len(result.away_goals_probs)) * result.away_goals_probs)

    assert expected_home_goals > expected_away_goals


def test_implied_poisson_goals_invalid_input():
    """Test with invalid probability values."""
    invalid_proba = ProbaResult(
        proba_home=1.2,
        proba_draw=0.2,
        proba_away=0.1,  # Invalid probability > 1
    )

    with pytest.raises(ArithmeticError):
        implied_poisson_goals(invalid_proba)


def test_implied_poisson_goals_equal_teams():
    """Test with equal team strengths."""
    equal_proba = ProbaResult(proba_home=0.35, proba_draw=0.30, proba_away=0.35)

    result = implied_poisson_goals(equal_proba)

    # Expected goals should be similar for both teams
    expected_home_goals = np.sum(np.arange(len(result.home_goals_probs)) * result.home_goals_probs)
    expected_away_goals = np.sum(np.arange(len(result.away_goals_probs)) * result.away_goals_probs)

    assert np.isclose(expected_home_goals, expected_away_goals, atol=0.1)
