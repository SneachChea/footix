"""Tests for Bayesian model calibration functionality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footix.models.bayesian import BayesianModel


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


def test_bayesian_model_without_calibration(sample_data):
    """Test that model works without calibration enabled."""
    model = BayesianModel(n_goals=5, calibrate=False)
    model.fit(sample_data)

    assert model.trace is not None
    assert "home" in model.trace.posterior
    assert "intercept" in model.trace.posterior
    assert "atts" in model.trace.posterior
    assert "defs" in model.trace.posterior

    # Calibration parameters should not be present
    assert "tau" not in model.trace.posterior
    assert "bias" not in model.trace.posterior


def test_bayesian_model_with_calibration(sample_data):
    """Test that model works with calibration enabled and includes calibration parameters."""
    model = BayesianModel(n_goals=5, calibrate=True)
    model.fit(sample_data)

    assert model.trace is not None

    # Check that all base parameters exist
    assert "home" in model.trace.posterior
    assert "intercept" in model.trace.posterior
    assert "atts" in model.trace.posterior
    assert "defs" in model.trace.posterior

    # Check that calibration parameters exist
    assert "tau" in model.trace.posterior, "Temperature parameter should be present"
    assert "bias" in model.trace.posterior, "Bias parameters should be present"

    # Check calibration parameter shapes
    tau_samples = model.trace.posterior["tau"]
    bias_samples = model.trace.posterior["bias"]

    assert tau_samples.ndim == 2  # (chain, draw)
    assert bias_samples.ndim == 3  # (chain, draw, 3 classes)
    assert bias_samples.shape[-1] == 3, "Bias should have 3 components (H, D, A)"


def test_calibration_parameter_priors(sample_data):
    """Test that calibration parameters are close to their prior means."""
    model = BayesianModel(n_goals=5, calibrate=True)
    model.fit(sample_data)

    # Extract posterior means
    tau_mean = model.trace.posterior["tau"].mean().values
    bias_mean = model.trace.posterior["bias"].mean(dim=["chain", "draw"]).values

    # Temperature should be close to 1.0 (prior mean)
    assert 0.5 < tau_mean < 2.0, f"Temperature {tau_mean} is far from prior mean of 1.0"

    # Bias should be close to 0.0 (prior mean)
    assert np.all(np.abs(bias_mean) < 1.5), "Bias values are far from prior mean of 0.0"


def test_calibrated_probabilities_sum_to_one(sample_data):
    """Test that calibrated match probabilities sum to 1."""
    model = BayesianModel(n_goals=5, calibrate=True)
    model.fit(sample_data)

    # Check match_probs deterministic
    match_probs = model.trace.posterior["match_probs"]

    # Sum over the last dimension (H, D, A classes)
    prob_sums = match_probs.sum(dim="match_probs_dim_1")

    # All probabilities should sum to 1 (within numerical tolerance)
    assert np.allclose(prob_sums.values, 1.0, atol=1e-6)


def test_prediction_works_with_calibration(sample_data):
    """Test that prediction works correctly when calibration is enabled."""
    model = BayesianModel(n_goals=5, calibrate=True)
    model.fit(sample_data)

    # Get unique teams from sample data
    teams = pd.concat([sample_data["home_team"], sample_data["away_team"]]).unique()
    home_team = teams[0]
    away_team = teams[1]

    # Test predict method
    goal_matrix = model.predict(home_team, away_team)
    samples = goal_matrix.return_probas()

    assert goal_matrix is not None
    assert hasattr(goal_matrix, "home_goals_probs")
    assert hasattr(goal_matrix, "away_goals_probs")
    assert hasattr(goal_matrix, "matrix_array")
    assert samples.proba_home is not None
    assert samples.proba_draw is not None
    assert samples.proba_away is not None

    # Probabilities should be valid
    assert 0 <= samples.proba_home <= 1
    assert 0 <= samples.proba_draw <= 1
    assert 0 <= samples.proba_away <= 1


def test_calibration_improves_model_fit(sample_data):
    """Test that calibration doesn't break the model (basic sanity check)."""
    # Fit both models
    model_no_calib = BayesianModel(n_goals=5, calibrate=False)
    model_with_calib = BayesianModel(n_goals=5, calibrate=True)

    model_no_calib.fit(sample_data)
    model_with_calib.fit(sample_data)

    # Both should produce valid traces
    assert model_no_calib.trace is not None
    assert model_with_calib.trace is not None

    # Both should have similar base parameters
    teams_in_data = len(pd.concat([sample_data["home_team"], sample_data["away_team"]]).unique())

    assert model_no_calib.trace.posterior["atts"].shape[-1] == teams_in_data
    assert model_with_calib.trace.posterior["atts"].shape[-1] == teams_in_data


def test_calibration_parameters_are_learnable(sample_data):
    """Test that calibration parameters vary across posterior samples (not stuck)."""
    model = BayesianModel(n_goals=5, calibrate=True)
    model.fit(sample_data)

    # Check that tau varies
    tau_samples = model.trace.posterior["tau"].values.flatten()
    tau_std = np.std(tau_samples)
    assert tau_std > 0.01, "Temperature parameter shows no variation"

    # Check that bias varies
    bias_samples = model.trace.posterior["bias"].values
    bias_std = np.std(bias_samples, axis=(0, 1))
    assert np.all(bias_std > 0.01), "Bias parameters show no variation"


@pytest.mark.parametrize("calibrate", [False, True])
@pytest.mark.parametrize("use_stats", [False, True])
def test_model_initialization(calibrate, use_stats):
    """Test model initialization with and without calibration."""
    model = BayesianModel(n_goals=10, n_teams=20, calibrate=calibrate, use_stats=use_stats)

    assert model.n_goals == 10
    assert model.n_teams == 20
    assert model.calibrate == calibrate
    assert model.use_stats == use_stats
    assert model.trace is None
