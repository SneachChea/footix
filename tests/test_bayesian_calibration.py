"""Tests for Bayesian model and out-of-sample 1X2 calibration."""

from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from footix.models.bayesian import BayesianModel
from footix.models.calibration import OutcomeCalibrator
from footix.models.score_matrix import GoalMatrix

pytestmark = pytest.mark.bayesian


def _build_fake_trace(n_teams: int, n_matches: int) -> az.InferenceData:
    """Create a lightweight posterior object for deterministic model tests."""
    draws = 8
    draw_axis = np.arange(draws, dtype=float)

    home = np.full((1, draws, n_teams), 0.1, dtype=float)
    intercept = np.linspace(0.2, 0.35, draws, dtype=float).reshape(1, draws)

    atts_base = np.linspace(-0.15, 0.15, n_teams, dtype=float)
    defs_base = np.linspace(0.12, -0.12, n_teams, dtype=float)
    atts = (atts_base.reshape(1, n_teams) + 0.01 * draw_axis.reshape(draws, 1))[None, :, :]
    defs = (defs_base.reshape(1, n_teams) + 0.005 * draw_axis.reshape(draws, 1))[None, :, :]

    posterior: dict[str, np.ndarray] = {
        "home": home,
        "intercept": intercept,
        "atts": atts,
        "defs": defs,
    }

    return az.from_dict({"posterior": posterior})


def _patch_hierarchical_bayes(monkeypatch: Any, sample_data: pd.DataFrame) -> None:
    """Replace expensive MCMC with a deterministic fake posterior for tests."""

    def fake_hierarchical_bayes(
        self: BayesianModel,
        goals_home_obs: np.ndarray,
        goals_away_obs: np.ndarray,
        home_team: np.ndarray,
        away_team: np.ndarray,
        optional_stats: dict[str, Any] | None = None,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> az.InferenceData:
        _ = goals_home_obs, goals_away_obs, home_team, away_team, optional_stats, sample_kwargs
        return _build_fake_trace(
            n_teams=int(self.n_teams or 0),
            n_matches=len(sample_data),
        )

    monkeypatch.setattr(BayesianModel, "hierarchical_bayes", fake_hierarchical_bayes)


@pytest.fixture
def sample_data():
    """Generate sample football match data for testing.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: home_team, away_team, fthg, ftag

    """
    np.random.seed(42)
    teams = ["Team_A", "Team_B", "Team_C", "Team_D"]
    n_matches = 30

    home_teams = np.random.choice(teams, size=n_matches)
    away_teams = np.random.choice(teams, size=n_matches)

    # Ensure no team plays itself
    mask = home_teams == away_teams
    while mask.any():
        away_teams[mask] = np.random.choice(teams, size=mask.sum())
        mask = home_teams == away_teams

    # Generate goals with some realistic patterns
    home_goals = np.random.poisson(lam=1.5, size=n_matches)
    away_goals = np.random.poisson(lam=1.2, size=n_matches)

    return pd.DataFrame(
        {
            "home_team": home_teams,
            "away_team": away_teams,
            "fthg": home_goals,
            "ftag": away_goals,
        }
    )


def test_bayesian_model_fits_without_calibration(sample_data, monkeypatch: Any):
    """Test that the model fits and exposes the base posterior parameters."""
    _patch_hierarchical_bayes(monkeypatch, sample_data)
    model = BayesianModel(n_goals=5)
    model.fit(sample_data)

    assert model.trace is not None
    assert "home" in model.trace.posterior
    assert "intercept" in model.trace.posterior
    assert "atts" in model.trace.posterior
    assert "defs" in model.trace.posterior


def test_team_name_mapping_uses_sorted_names(sample_data, monkeypatch: Any):
    """Map team names to the same deterministic order used by LabelEncoder."""
    _patch_hierarchical_bayes(monkeypatch, sample_data)
    model = BayesianModel(n_goals=5)
    model.fit(sample_data)

    teams = sorted(set(sample_data["home_team"]) | set(sample_data["away_team"]))
    assert model._team_to_id == {team: index for index, team in enumerate(teams)}


def test_numeric_team_ids_are_encoded(monkeypatch: Any):
    """Preserve numeric team identifiers while encoding them for the model."""
    sample_data = pd.DataFrame(
        {
            "home_team": [1, 2],
            "away_team": [2, 1],
            "fthg": [1, 0],
            "ftag": [0, 1],
        }
    )

    def fake_hierarchical_bayes(
        self: BayesianModel,
        goals_home_obs: np.ndarray,
        goals_away_obs: np.ndarray,
        home_team: np.ndarray,
        away_team: np.ndarray,
        optional_stats: dict[str, Any] | None = None,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> az.InferenceData:
        _ = goals_home_obs, goals_away_obs, optional_stats, sample_kwargs
        assert np.array_equal(home_team, [0, 1])
        assert np.array_equal(away_team, [1, 0])
        return _build_fake_trace(n_teams=2, n_matches=len(sample_data))

    monkeypatch.setattr(BayesianModel, "hierarchical_bayes", fake_hierarchical_bayes)
    model = BayesianModel(n_goals=5)
    model.fit(sample_data)

    assert model._team_to_id == {1: 0, 2: 1}


def test_posterior_samples_cover_total_goals(sample_data, monkeypatch: Any):
    """Posterior samples retain the U2.5/O2.5 ordering used by GoalMatrix."""
    _patch_hierarchical_bayes(monkeypatch, sample_data)
    model = BayesianModel(n_goals=5, random_seed=7)
    model.fit(sample_data)

    teams = sorted(set(sample_data["home_team"]) | set(sample_data["away_team"]))
    samples = model.get_market_samples(teams[0], teams[1], "U2.5")
    home_id = model._team_to_id[teams[0]]
    away_id = model._team_to_id[teams[1]]
    posterior = model.trace.posterior  # type: ignore[union-attr]
    home = posterior["home"].stack(sample=("chain", "draw")).values
    atts = posterior["atts"].stack(sample=("chain", "draw")).values
    defs = posterior["defs"].stack(sample=("chain", "draw")).values
    intercept = posterior["intercept"].stack(sample=("chain", "draw")).values
    goals = np.arange(model.n_goals)
    expected_under = []
    for index in range(intercept.size):
        mu_home = np.exp(
            intercept[index] + home[home_id, index] + atts[home_id, index] + defs[away_id, index]
        )
        mu_away = np.exp(intercept[index] + atts[away_id, index] + defs[home_id, index])
        expected_under.append(
            GoalMatrix(
                stats.poisson.pmf(goals, mu_home), stats.poisson.pmf(goals, mu_away)
            ).less_25_goals()
        )

    assert samples.ndim == 2
    assert samples.shape[1] == 2
    assert np.allclose(samples.sum(axis=1), 1.0)
    assert np.allclose(samples[:, 0], expected_under)
    assert np.allclose(samples[:, 1], 1.0 - np.asarray(expected_under))


def test_posterior_samples_are_valid_1x2_probabilities(sample_data, monkeypatch: Any):
    """1X2 posterior samples are per-draw probabilities summing to one."""
    _patch_hierarchical_bayes(monkeypatch, sample_data)
    model = BayesianModel(n_goals=5)
    model.fit(sample_data)

    teams = sorted(set(sample_data["home_team"]) | set(sample_data["away_team"]))
    samples = model.get_market_samples(teams[0], teams[1], "1X2")

    assert samples.ndim == 2
    assert samples.shape[1] == 3
    assert np.allclose(samples.sum(axis=1), 1.0)
    assert np.all((samples >= 0) & (samples <= 1))


def test_posterior_samples_batch_matches_per_match_calls(sample_data, monkeypatch: Any):
    """The batch sampler and the per-match sampler share the draw axis."""
    _patch_hierarchical_bayes(monkeypatch, sample_data)
    model = BayesianModel(n_goals=5)
    model.fit(sample_data)

    teams = sorted(set(sample_data["home_team"]) | set(sample_data["away_team"]))
    pairs = [(teams[0], teams[1]), (teams[2], teams[3])]
    batch = model.get_market_samples_batch(pairs, "1X2")
    single = model.get_market_samples(teams[0], teams[1], "1X2")

    assert batch.shape == (8, 2, 3)
    assert np.allclose(batch[:, 0, :], single)


@pytest.mark.parametrize("use_stats", [False, True])
def test_model_initialization(use_stats):
    """Test model initialization."""
    model = BayesianModel(n_goals=10, n_teams=20, use_stats=use_stats)

    assert model.n_goals == 10
    assert model.n_teams == 20
    assert model.use_stats == use_stats
    assert model.trace is None


def test_calibrator_identity_before_warmup():
    """The calibrator is the exact identity before enough observations."""
    calibrator = OutcomeCalibrator(warmup=10)
    calibrator.accumulate(np.asarray([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]]), [0, 1])
    calibrator.fit()

    assert calibrator.tau == 1.0
    assert np.allclose(calibrator.bias, 0.0)
    probs = np.asarray([0.7, 0.2, 0.1])
    assert np.allclose(calibrator.apply(probs), probs)


def test_calibrator_learns_on_biased_model():
    """On an overconfident model, calibration pulls probabilities toward reality."""
    rng = np.random.default_rng(0)
    n = 300
    probs = np.tile([0.8, 0.1, 0.1], (n, 1))
    outcomes = rng.choice(3, size=n, p=[0.6, 0.2, 0.2])

    calibrator = OutcomeCalibrator(warmup=0, reg=1e-2)
    calibrator.accumulate(probs, outcomes)
    calibrator.fit()

    calibrated = calibrator.apply([0.8, 0.1, 0.1])
    assert not np.allclose(calibrated, [0.8, 0.1, 0.1])
    assert calibrated[0] < 0.8  # overconfident home win pulled down

    # Out-of-sample log-loss improves on a held-out sample
    held_out = rng.choice(3, size=200, p=[0.6, 0.2, 0.2])
    raw_ll = -np.log([0.8, 0.1, 0.1])[held_out].mean()
    calibrated_ll = -np.log(calibrator.apply(probs[:200])[np.arange(200), held_out]).mean()
    assert calibrated_ll < raw_ll


def test_calibrator_sums_to_one():
    """Calibrated probabilities are normalized whatever the input shape."""
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(3), size=100)
    outcomes = rng.choice(3, size=100)

    calibrator = OutcomeCalibrator(warmup=0)
    calibrator.accumulate(probs, outcomes)
    calibrator.fit()

    assert np.allclose(calibrator.apply(probs).sum(axis=1), 1.0)
    assert np.isclose(calibrator.apply(probs[0]).sum(), 1.0)


def test_calibrator_apply_broadcasts_over_draws():
    """Point and draw application of the calibrator are the same transform."""
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.ones(3), size=50)
    outcomes = rng.choice(3, size=50)

    calibrator = OutcomeCalibrator(warmup=10)
    calibrator.accumulate(probs, outcomes)
    calibrator.fit()

    draws = rng.dirichlet(np.ones(3), size=24)
    applied = calibrator.apply(draws)
    expected = np.stack([calibrator.apply(draw) for draw in draws])
    assert np.allclose(applied, expected)


def test_calibrator_accumulate_validates_shapes():
    """One outcome per probability row is enforced."""
    calibrator = OutcomeCalibrator()
    with pytest.raises(ValueError, match="outcomes"):
        calibrator.accumulate(np.ones((5, 3)), [0, 1])
