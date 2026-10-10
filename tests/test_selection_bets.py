import numpy as np
import pytest

from footix import SampleProbaResult
from footix.strategy.bets import Bet, OddsInput
from footix.strategy.select_bets import (
    EdgeFloorConfig,
    OddsRange,
    select_bets_diagnostics,
    select_bets_posterior,
    select_matches_posterior,
)


@pytest.fixture
def sample_odds_input():
    """Create sample odds input for testing."""
    return [
        OddsInput(home_team="Team1", away_team="Team2", odds=[2.0, 3.0, 4.0]),
        OddsInput(home_team="Team3", away_team="Team4", odds=[1.5, 4.0, 3.0]),
    ]


@pytest.fixture
def sample_lambda_samples():
    return {
        "Team1 - Team2": SampleProbaResult(
            proba_home=np.full(1000, 0.9),
            proba_draw=np.full(1000, 0.05),
            proba_away=np.full(1000, 0.05),
        ),
        "Team3 - Team4": SampleProbaResult(
            proba_home=np.full(1000, 0.1),
            proba_draw=np.full(1000, 0.50),
            proba_away=np.full(1000, 0.4),
        ),
    }


def test_select_matches_single_bet():
    """Test selecting single bet per match with default thresholds."""
    odds_input = [OddsInput("Home", "Away", [2.0, 3.0, 4.0])]

    # Create samples that will favor home win
    # Calculate home, draw, and away probabilities from lambda values
    proba_home = np.full(1000, 0.6)
    proba_draw = np.full(1000, 0.3)
    proba_away = np.full(1000, 0.1)
    lambda_samples = {"Home - Away": SampleProbaResult(proba_home, proba_draw, proba_away)}

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
    lambda_samples = {
        "Home - Away": SampleProbaResult(
            proba_home=np.full(1000, 0.8),
            proba_draw=np.full(1000, 0.1),
            proba_away=np.full(1000, 0.1),
        )
    }
    config = EdgeFloorConfig(default_edge_floor=0.9)
    selected = select_matches_posterior(
        odds_input=odds_input,
        lambda_samples=lambda_samples,
        config=config,
    )

    assert len(selected) == 0


def test_odds_range_config():
    """Test that EdgeFloorConfig properly applies different thresholds for different odds
    ranges.
    """
    odds_input = [
        OddsInput("Home1", "Away1", [1.5, 3.0, 4.0]),  # Low odds
        OddsInput("Home2", "Away2", [3.0, 3.5, 4.0]),  # Medium odds
    ]

    # Create sample probabilities
    lambda_samples = {
        "Home1 - Away1": SampleProbaResult(
            np.full(1000, 0.6), np.full(1000, 0.3), np.full(1000, 0.1)
        ),
        "Home2 - Away2": SampleProbaResult(
            np.full(1000, 0.6), np.full(1000, 0.3), np.full(1000, 0.1)
        ),
    }

    config = EdgeFloorConfig(
        ranges=[
            OddsRange(min_odds=1.0, max_odds=2.0, edge=0.2, prob_edge=0.6),  # Strict for low odds
            OddsRange(min_odds=2.0, max_odds=5.0, edge=0.1, prob_edge=0.55),
        ]
    )

    selected = select_matches_posterior(
        odds_input=odds_input,
        lambda_samples=lambda_samples,
        config=config,
    )

    # The second match (medium odds) should be selected due to lower threshold
    assert len(selected) == 1
    assert selected[0].match_id == "Home2 - Away2"


def test_prob_threshold_filtering():
    """Test that probability threshold properly filters bets."""
    odds_input = [OddsInput("Home", "Away", [2.0, 3.0, 4.0])]

    # Create noisy samples to get uncertain edges
    rng = np.random.default_rng(42)  # For reproducibility
    lambda_samples = {
        "Home - Away": SampleProbaResult(
            rng.normal(0.4, 0.2, 1000), rng.normal(0.3, 0.2, 1000), rng.normal(0.3, 0.2, 1000)
        )
    }

    config = EdgeFloorConfig(default_prob_edge=0.99)  # Very high certainty requirement
    selected = select_matches_posterior(
        odds_input=odds_input,
        lambda_samples=lambda_samples,
        config=config,
    )

    assert len(selected) == 0


def test_sorting_by_edge():
    """Test that bets are sorted by descending edge."""
    odds_input = [
        OddsInput("Home1", "Away1", [2.0, 3.0, 4.0]),
        OddsInput("Home2", "Away2", [4.0, 3.0, 2.0]),
    ]

    lambda_samples = {
        "Home1 - Away1": SampleProbaResult(
            np.full(1000, 0.6),
            np.full(1000, 0.3),
            np.full(1000, 0.1),  # Home favored
        ),
        "Home2 - Away2": SampleProbaResult(
            np.full(1000, 0.1),
            np.full(1000, 0.3),
            np.full(1000, 0.6),  # Away favored
        ),
    }

    selected = select_matches_posterior(
        odds_input=odds_input, lambda_samples=lambda_samples, single_bet_per_game=False
    )

    # Check sorting
    for i in range(len(selected) - 1):
        assert selected[i].edge_mean >= selected[i + 1].edge_mean


def test_empty_inputs():
    """Test behavior with empty inputs."""
    config = EdgeFloorConfig(
        ranges=[OddsRange(min_odds=1.5, max_odds=3.0, edge=0.1)], default_edge_floor=0.2
    )
    selected = select_matches_posterior(
        odds_input=[],
        lambda_samples={},
        config=config,
    )

    assert len(selected) == 0


def test_invalid_match_id():
    """Test error handling for missing match ID in lambda samples."""
    odds_input = [OddsInput("Home", "Away", [2.0, 3.0, 4.0])]

    lambda_samples = {}  # Empty samples

    with pytest.raises(KeyError):
        select_matches_posterior(odds_input=odds_input, lambda_samples=lambda_samples)


def _norm_samples(mean: float, std: float, seed: int = 0, n: int = 10_000) -> np.ndarray:
    """Posterior probability samples with controlled dispersion."""
    return np.clip(np.random.default_rng(seed).normal(mean, std, n), 1e-6, 1.0)


def test_select_posterior_rejects_big_favorite() -> None:
    """A heavy favorite with a negative conservative edge is rejected."""
    bet = Bet("m1", "H", 1.35, 0.78)
    samples = {("m1", "H"): _norm_samples(0.78, 0.08)}

    selected = select_bets_posterior([bet], samples)

    assert selected == []


def test_select_posterior_keeps_value_bet() -> None:
    """A bet whose 10% lower edge bound is positive is kept."""
    bet = Bet("m1", "H", 1.80, 0.64)
    samples = {("m1", "H"): _norm_samples(0.64, 0.03)}

    selected = select_bets_posterior([bet], samples)

    assert len(selected) == 1
    assert selected[0] == bet


def test_select_posterior_picks_best_robust_kelly_per_match() -> None:
    """Within a match the highest robust Kelly score wins."""
    home = Bet("m1", "H", 2.0, 0.65)
    away = Bet("m1", "A", 4.0, 0.30)
    samples = {
        ("m1", "H"): _norm_samples(0.65, 0.03, seed=1),
        ("m1", "A"): _norm_samples(0.30, 0.03, seed=2),
    }

    selected = select_bets_posterior([home, away], samples)

    assert len(selected) == 1
    assert selected[0].market == "H"


def test_select_posterior_ignores_ambiguous_match() -> None:
    """A match with two near-identical selections is left alone."""
    home = Bet("m1", "H", 2.0, 0.50)
    away = Bet("m1", "A", 2.0, 0.50)
    samples = {
        ("m1", "H"): _norm_samples(0.50, 0.04, seed=3),
        ("m1", "A"): _norm_samples(0.50, 0.04, seed=4),
    }

    selected = select_bets_posterior([home, away], samples, rho_min=0.60)

    assert selected == []


def test_select_posterior_requires_samples() -> None:
    bet = Bet("m1", "H", 2.0, 0.60)

    with pytest.raises(KeyError):
        select_bets_posterior([bet], {})


def test_select_posterior_rejects_canonical_market_as_selection() -> None:
    """'1X2'/'O/U2.5' as market would collapse distinct selections onto one key."""
    bet = Bet("m1", "1X2", 2.0, 0.60)
    samples = {("m1", "H"): _norm_samples(0.60, 0.03)}

    with pytest.raises(ValueError, match="selection"):
        select_bets_posterior([bet], samples)


def test_select_posterior_rejects_duplicate_keys() -> None:
    first = Bet("m1", "H", 2.0, 0.60)
    duplicate = Bet("m1", "H", 2.2, 0.60)
    samples = {("m1", "H"): _norm_samples(0.60, 0.03)}

    with pytest.raises(ValueError, match="Duplicate"):
        select_bets_posterior([first, duplicate], samples)


def test_select_posterior_rejects_misaligned_draws() -> None:
    """Comparisons inside a match are draw by draw: lengths must match."""
    home = Bet("m1", "H", 2.0, 0.60)
    away = Bet("m1", "A", 2.0, 0.30)
    samples = {
        ("m1", "H"): _norm_samples(0.60, 0.03, n=10_000),
        ("m1", "A"): _norm_samples(0.30, 0.03, n=5_000),
    }

    with pytest.raises(ValueError, match="same draw axis"):
        select_bets_posterior([home, away], samples)


def test_select_diagnostics_reports_rejection_reasons() -> None:
    favorite = Bet("m1", "H", 1.35, 0.78)
    value = Bet("m2", "H", 1.80, 0.64)
    amb_home = Bet("m3", "H", 2.0, 0.55)
    amb_away = Bet("m3", "A", 2.0, 0.55)
    samples = {
        ("m1", "H"): _norm_samples(0.78, 0.08, seed=5),
        ("m2", "H"): _norm_samples(0.64, 0.03, seed=6),
        ("m3", "H"): _norm_samples(0.55, 0.02, seed=7),
        ("m3", "A"): _norm_samples(0.55, 0.02, seed=8),
    }

    report = select_bets_diagnostics([favorite, value, amb_home, amb_away], samples, rho_min=0.60)

    assert list(report.columns) == [
        "match_id",
        "market",
        "selection",
        "q_edge",
        "mean_edge",
        "prob_edge_positive",
        "robust_kelly",
        "rho",
        "passed_edge_filter",
        "passed_ambiguity_filter",
        "rejection_reason",
    ]
    report = report.set_index(["match_id", "selection"])
    assert report.loc[("m1", "H"), "rejection_reason"] == "edge_bound"
    assert report.loc[("m2", "H"), "rejection_reason"] is None
    reasons_m3 = sorted(str(r) for r in report.loc["m3"]["rejection_reason"])
    assert reasons_m3 == ["ambiguity", "not_best"]
