import math

import numpy as np
import pytest

from footix.models.score_matrix import GoalMatrix
from footix.utils.typing import ProbaResult


class TestGoalMatrixInitialization:
    """Tests for GoalMatrix initialization and validation."""

    def test_basic_initialization(self):
        """Test basic initialization with valid inputs."""
        home_probs = [0.1, 0.3, 0.4, 0.2]
        away_probs = [0.2, 0.3, 0.35, 0.15]
        gm = GoalMatrix(home_probs, away_probs)

        assert isinstance(gm.matrix_array, np.ndarray)
        assert gm.matrix_array.shape == (4, 4)
        assert gm.correlation_matrix is None

    def test_initialization_with_correlation(self):
        """Test initialization with correlation matrix."""
        home_probs = [0.1, 0.3, 0.4, 0.2]
        away_probs = [0.2, 0.3, 0.35, 0.15]
        corr_matrix = np.eye(4)
        gm = GoalMatrix(home_probs, away_probs, correlation_matrix=corr_matrix)

        assert gm.correlation_matrix is not None
        assert gm.matrix_array.shape == (4, 4)

    def test_initialization_converts_to_array(self):
        """Test that inputs are converted to numpy arrays."""
        home_probs = [0.2, 0.5, 0.3]
        away_probs = [0.3, 0.4, 0.3]
        gm = GoalMatrix(home_probs, away_probs)

        assert isinstance(gm.home_goals_probs, np.ndarray)
        assert isinstance(gm.away_goals_probs, np.ndarray)

    def test_mismatched_array_lengths(self):
        """Test that mismatched array lengths raise ValueError."""
        home_probs = [0.2, 0.5, 0.3]
        away_probs = [0.3, 0.4]

        with pytest.raises(
            ValueError, match="home_goals_probs and away_goals_probs must have the same length"
        ):
            GoalMatrix(home_probs, away_probs)

    def test_multidimensional_home_probs(self):
        """Test that multidimensional home_probs raises ValueError."""
        home_probs = [[0.2, 0.5], [0.3, 0.2]]
        away_probs = [0.3, 0.4, 0.3]

        with pytest.raises(ValueError, match="must be 1D arrays"):
            GoalMatrix(home_probs, away_probs)

    def test_multidimensional_away_probs(self):
        """Test that multidimensional away_probs raises ValueError."""
        home_probs = [0.2, 0.5, 0.3]
        away_probs = [[0.3, 0.4], [0.2, 0.1]]

        with pytest.raises(ValueError, match="must be 1D arrays"):
            GoalMatrix(home_probs, away_probs)

    def test_mismatched_correlation_matrix_size(self):
        """Test that correlation matrix size mismatch raises ValueError."""
        home_probs = [0.2, 0.5, 0.3]
        away_probs = [0.3, 0.4, 0.3]
        corr_matrix = np.eye(4)

        with pytest.raises(
            ValueError,
            match="correlation_matrix must have shape \(n, n\) matching probabilities length",
        ):
            GoalMatrix(home_probs, away_probs, correlation_matrix=corr_matrix)

    def test_initialization_normalizes_inputs(self):
        """Test that probability vectors are normalized to sum to 1."""
        home_probs = [1.0, 1.0, 2.0]
        away_probs = [2.0, 1.0, 1.0]
        gm = GoalMatrix(home_probs, away_probs)

        assert math.isclose(float(np.sum(gm.home_goals_probs)), 1.0, rel_tol=1e-12)
        assert math.isclose(float(np.sum(gm.away_goals_probs)), 1.0, rel_tol=1e-12)
        assert math.isclose(float(np.sum(gm.matrix_array)), 1.0, rel_tol=1e-12)

    def test_negative_probabilities_raise(self):
        """Test that negative probabilities are rejected."""
        home_probs = [-0.1, 1.1]
        away_probs = [0.5, 0.5]
        with pytest.raises(ValueError, match="home_goals_probs must be non-negative"):
            GoalMatrix(home_probs, away_probs)

    def test_nan_probabilities_raise(self):
        """Test that NaN values are rejected."""
        home_probs = [np.nan, 1.0]
        away_probs = [0.5, 0.5]
        with pytest.raises(ValueError, match="home_goals_probs must contain only finite values"):
            GoalMatrix(home_probs, away_probs)

    def test_zero_mass_probabilities_raise(self):
        """Test that zero total probability mass is rejected."""
        home_probs = [0.0, 0.0]
        away_probs = [0.5, 0.5]
        with pytest.raises(
            ValueError, match="home_goals_probs must have positive total probability mass"
        ):
            GoalMatrix(home_probs, away_probs)

    def test_invalid_correlation_matrix_ndim_raise(self):
        """Test that non-2D correlation matrices are rejected."""
        home_probs = [0.5, 0.5]
        away_probs = [0.5, 0.5]
        corr_matrix = np.zeros((2, 2, 1))
        with pytest.raises(ValueError, match="correlation_matrix must be a 2D array"):
            GoalMatrix(home_probs, away_probs, correlation_matrix=corr_matrix)

    def test_negative_correlation_matrix_raise(self):
        """Test that negative values in correlation_matrix are rejected."""
        home_probs = [0.5, 0.5]
        away_probs = [0.5, 0.5]
        corr_matrix = np.array([[1.0, -1.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="correlation_matrix must be non-negative"):
            GoalMatrix(home_probs, away_probs, correlation_matrix=corr_matrix)

    def test_non_finite_correlation_matrix_raise(self):
        """Test that non-finite values in correlation_matrix are rejected."""
        home_probs = [0.5, 0.5]
        away_probs = [0.5, 0.5]
        corr_matrix = np.array([[1.0, np.nan], [1.0, 1.0]])
        with pytest.raises(ValueError, match="correlation_matrix must contain only finite values"):
            GoalMatrix(home_probs, away_probs, correlation_matrix=corr_matrix)


class TestReturnProbas:
    """Tests for return_probas method."""

    def test_return_probas_basic(self):
        """Test basic probability calculation."""
        home_probs = [0.5, 0.5]
        away_probs = [0.5, 0.5]
        gm = GoalMatrix(home_probs, away_probs)

        result = gm.return_probas()

        assert isinstance(result, ProbaResult)
        assert math.isclose(result.proba_home + result.proba_draw + result.proba_away, 1.0)

    def test_return_probas_values(self):
        """Test that probabilities are correctly calculated."""
        home_probs = np.array([0.6, 0.4])
        away_probs = np.array([0.6, 0.4])
        gm = GoalMatrix(home_probs, away_probs)

        result = gm.return_probas()

        # Draw: matrix[0,0] + matrix[1,1] = 0.36 + 0.16 = 0.52
        # Home win: matrix[1,0] = 0.24
        # Away win: matrix[0,1] = 0.24
        expected_draw = 0.36 + 0.16
        assert math.isclose(result.proba_draw, expected_draw, rel_tol=1e-5)

    def test_return_probas_with_correlation(self):
        """Test probability calculation with correlation matrix."""
        home_probs = [0.5, 0.5]
        away_probs = [0.5, 0.5]
        corr_matrix = np.eye(2)
        gm = GoalMatrix(home_probs, away_probs, correlation_matrix=corr_matrix)

        result = gm.return_probas()

        assert math.isclose(
            result.proba_home + result.proba_draw + result.proba_away, 1.0, rel_tol=1e-5
        )


class TestGoalMarketMethods:
    """Tests for under/over goal methods."""

    def test_less_15_goals(self):
        """Test less than 1.5 goals calculation."""
        home_probs = [0.3, 0.3, 0.2, 0.2]
        away_probs = [0.3, 0.3, 0.2, 0.2]
        gm = GoalMatrix(home_probs, away_probs)

        result = gm.less_15_goals()

        # Should be: (0,0) + (0,1) + (1,0)
        expected = gm.matrix_array[0, 0] + gm.matrix_array[0, 1] + gm.matrix_array[1, 0]
        assert math.isclose(result, expected)

    def test_less_15_goals_insufficient_length(self):
        """Test that less_15_goals raises TypeError with short arrays."""
        home_probs = [0.5]
        away_probs = [0.5]
        gm = GoalMatrix(home_probs, away_probs)

        with pytest.raises(
            ValueError, match="must have length >= 2 for less_15_goals"
        ):
            gm.less_15_goals()

    def test_less_25_goals(self):
        """Test less than 2.5 goals calculation."""
        home_probs = [0.3, 0.3, 0.2, 0.2]
        away_probs = [0.3, 0.3, 0.2, 0.2]
        gm = GoalMatrix(home_probs, away_probs)

        result = gm.less_25_goals()

        # Should include all cells up to 2 goals for each team
        assert result >= 0 and result <= 1

    def test_less_25_goals_insufficient_length(self):
        """Test that less_25_goals raises TypeError with short arrays."""
        home_probs = [0.5, 0.5]
        away_probs = [0.5, 0.5]
        gm = GoalMatrix(home_probs, away_probs)

        with pytest.raises(
            ValueError, match="must have length >= 3 for less_25_goals"
        ):
            gm.less_25_goals()

    def test_more_15_goals(self):
        """Test more than 1.5 goals calculation."""
        home_probs = [0.3, 0.3, 0.2, 0.2]
        away_probs = [0.3, 0.3, 0.2, 0.2]
        gm = GoalMatrix(home_probs, away_probs)

        result = gm.more_15_goals()
        less_result = gm.less_15_goals()

        assert math.isclose(result + less_result, 1.0, rel_tol=1e-5)

    def test_more_25_goals(self):
        """Test more than 2.5 goals calculation."""
        home_probs = [0.3, 0.3, 0.2, 0.2]
        away_probs = [0.3, 0.3, 0.2, 0.2]
        gm = GoalMatrix(home_probs, away_probs)

        result = gm.more_25_goals()
        less_result = gm.less_25_goals()

        assert math.isclose(result + less_result, 1.0, rel_tol=1e-5)


class TestAsianHandicap:
    """Tests for asian_handicap_results method."""

    def test_asian_handicap_no_handicap(self):
        """Test Asian handicap with zero handicap (should equal return_probas)."""
        home_probs = [0.5, 0.5]
        away_probs = [0.5, 0.5]
        gm = GoalMatrix(home_probs, away_probs)

        ah_result = gm.asian_handicap_results(0)
        probas_result = gm.return_probas()

        assert math.isclose(ah_result.proba_home, probas_result.proba_home, rel_tol=1e-5)
        assert math.isclose(ah_result.proba_draw, probas_result.proba_draw, rel_tol=1e-5)
        assert math.isclose(ah_result.proba_away, probas_result.proba_away, rel_tol=1e-5)

    def test_asian_handicap_positive_handicap(self):
        """Test Asian handicap with positive handicap favoring home."""
        home_probs = [0.5, 0.5]
        away_probs = [0.5, 0.5]
        gm = GoalMatrix(home_probs, away_probs)

        ah_result = gm.asian_handicap_results(0.5)

        # Home should have higher probability with positive handicap
        assert math.isclose(
            ah_result.proba_home + ah_result.proba_draw + ah_result.proba_away, 1.0, rel_tol=1e-5
        )

    def test_asian_handicap_negative_handicap(self):
        """Test Asian handicap with negative handicap favoring away."""
        home_probs = [0.5, 0.5]
        away_probs = [0.5, 0.5]
        gm = GoalMatrix(home_probs, away_probs)

        ah_result = gm.asian_handicap_results(-0.5)

        assert math.isclose(
            ah_result.proba_home + ah_result.proba_draw + ah_result.proba_away, 1.0, rel_tol=1e-5
        )

    def test_asian_handicap_probabilities_sum(self):
        """Test that Asian handicap probabilities sum to 1."""
        home_probs = [0.3, 0.3, 0.2, 0.2]
        away_probs = [0.3, 0.3, 0.2, 0.2]
        gm = GoalMatrix(home_probs, away_probs)

        for handicap in [-1.5, -0.5, 0, 0.5, 1.5]:
            ah_result = gm.asian_handicap_results(handicap)
            total = ah_result.proba_home + ah_result.proba_draw + ah_result.proba_away
            assert math.isclose(total, 1.0, rel_tol=1e-5)


class TestDoubbleChance:
    """Tests for double_chance method."""

    def test_double_chance_basic(self):
        """Test basic double chance calculation."""
        home_probs = [0.5, 0.5]
        away_probs = [0.5, 0.5]
        gm = GoalMatrix(home_probs, away_probs)

        p_1x, p_x2, p_12 = gm.double_chance()

        # All should be between 0 and 1
        assert 0 <= p_1x <= 1
        assert 0 <= p_x2 <= 1
        assert 0 <= p_12 <= 1

    def test_double_chance_sum(self):
        """Test that double chance probabilities make sense."""
        home_probs = [0.3, 0.3, 0.2, 0.2]
        away_probs = [0.3, 0.3, 0.2, 0.2]
        gm = GoalMatrix(home_probs, away_probs)

        probas = gm.return_probas()
        p_1x, p_x2, p_12 = gm.double_chance()

        # Verify calculations
        assert math.isclose(p_1x, probas.proba_home + probas.proba_draw, rel_tol=1e-5)
        assert math.isclose(p_x2, probas.proba_draw + probas.proba_away, rel_tol=1e-5)
        assert math.isclose(p_12, probas.proba_home + probas.proba_away, rel_tol=1e-5)


class TestProbabilityBothTeamsScore:
    """Tests for probability_both_teams_scores method."""

    def test_probability_both_teams_scores_basic(self):
        """Test basic both teams to score calculation."""
        home_probs = [0.3, 0.3, 0.2, 0.2]
        away_probs = [0.3, 0.3, 0.2, 0.2]
        gm = GoalMatrix(home_probs, away_probs)

        result = gm.probability_both_teams_scores()

        # Should be between 0 and 1
        assert 0 <= result <= 1

    def test_probability_both_teams_scores_exclusion(self):
        """Test that 0-0 and shutout scores are excluded."""
        home_probs = [0.3, 0.3, 0.2, 0.2]
        away_probs = [0.3, 0.3, 0.2, 0.2]
        gm = GoalMatrix(home_probs, away_probs)

        result = gm.probability_both_teams_scores()

        # Should not include matrix[0,0] (0-0), matrix[0,1:]
        # (away scores, home 0), matrix[1:,0] (home scores, away 0)
        expected = np.sum(gm.matrix_array[1:, 1:])
        assert math.isclose(result, expected)


class TestGetProbableScore:
    """Tests for get_probable_score method."""

    def test_get_probable_score_basic(self):
        """Test getting the most probable score."""
        home_probs = [0.1, 0.5, 0.4]
        away_probs = [0.2, 0.6, 0.2]
        gm = GoalMatrix(home_probs, away_probs)

        score = gm.get_probable_score()

        assert isinstance(score, tuple)
        assert len(score) == 2
        assert isinstance(score[0], int)
        assert isinstance(score[1], int)

    def test_get_probable_score_matches_max(self):
        """Test that returned score matches the maximum in matrix."""
        home_probs = [0.1, 0.5, 0.4]
        away_probs = [0.2, 0.6, 0.2]
        gm = GoalMatrix(home_probs, away_probs)

        score = gm.get_probable_score()
        idx = np.unravel_index(np.argmax(gm.matrix_array), gm.matrix_array.shape)

        assert score[0] == idx[0]
        assert score[1] == idx[1]


class TestStringRepresentation:
    """Tests for __str__ method."""

    def test_str_representation(self):
        """Test string representation."""
        home_probs = [0.1, 0.3, 0.4, 0.2]
        away_probs = [0.2, 0.3, 0.35, 0.15]
        gm = GoalMatrix(home_probs, away_probs)

        result = str(gm)

        assert "Goal Matrix" in result
        assert "[" in result
        assert "]" in result
