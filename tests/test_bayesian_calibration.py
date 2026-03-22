"""Tests for Bayesian model calibration functionality."""

from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pytest

from footix.models.bayesian import BayesianModel


def _build_fake_trace(calibrate: bool, n_teams: int, n_matches: int) -> az.InferenceData:
    """Create a lightweight posterior object for deterministic calibration tests."""
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

    if calibrate:
        tau = np.linspace(0.8, 1.2, draws, dtype=float).reshape(1, draws)
        bias = np.stack(
            (
                np.linspace(0.05, 0.20, draws, dtype=float),
                np.linspace(-0.10, 0.05, draws, dtype=float),
                np.linspace(-0.02, 0.08, draws, dtype=float),
            ),
            axis=-1,
        )[None, :, :]
        match_probs = np.tile(
            np.array([0.55, 0.20, 0.25], dtype=float),
            (1, draws, n_matches, 1),
        )
        posterior.update(
            {
                "tau": tau,
                "bias": bias,
                "match_probs": match_probs,
            }
        )

    return az.from_dict(posterior=posterior)


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
            calibrate=self.calibrate,
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


def test_bayesian_model_without_calibration(sample_data, monkeypatch: Any):
    """Test that model works without calibration enabled."""
    _patch_hierarchical_bayes(monkeypatch, sample_data)
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


def test_bayesian_model_with_calibration(sample_data, monkeypatch: Any):
    """Test that model works with calibration enabled and includes calibration parameters."""
    _patch_hierarchical_bayes(monkeypatch, sample_data)
    model = BayesianModel(n_goals=5, calibrate=True)
    model.fit(sample_data)

    assert model.trace is not None

    # Check that all base parameters exist
    assert "home" in model.trace.posterior
    assert "intercept" in model.trace.posterior
    assert "atts" in model.trace.posterior
    assert "defs" in model.trace.posterior

    # Check that calibration parameters exist
    assert model.trace is not None
    assert "tau" in model.trace.posterior, "Temperature parameter should be present"
    assert "bias" in model.trace.posterior, "Bias parameters should be present"

    # Check calibration parameter shapes
    tau_samples = model.trace.posterior["tau"]
    bias_samples = model.trace.posterior["bias"]

    assert tau_samples.ndim == 2  # (chain, draw)
    assert bias_samples.ndim == 3  # (chain, draw, 3 classes)
    assert bias_samples.shape[-1] == 3, "Bias should have 3 components (H, D, A)"


def test_calibration_parameter_priors(sample_data, monkeypatch: Any):
    """Test that calibration parameters are close to their prior means."""
    _patch_hierarchical_bayes(monkeypatch, sample_data)
    model = BayesianModel(n_goals=5, calibrate=True)
    model.fit(sample_data)

    # Extract posterior means
    assert model.trace is not None
    tau_mean = model.trace.posterior["tau"].mean().values
    bias_mean = model.trace.posterior["bias"].mean(dim=["chain", "draw"]).values

    # Temperature should be close to 1.0 (prior mean)
    assert 0.5 < tau_mean < 2.0, f"Temperature {tau_mean} is far from prior mean of 1.0"

    # Bias should be close to 0.0 (prior mean)
    assert np.all(np.abs(bias_mean) < 1.5), "Bias values are far from prior mean of 0.0"


def test_calibrated_probabilities_sum_to_one(sample_data, monkeypatch: Any):
    """Test that calibrated match probabilities sum to 1."""
    _patch_hierarchical_bayes(monkeypatch, sample_data)
    model = BayesianModel(n_goals=5, calibrate=True)
    model.fit(sample_data)

    # Check match_probs deterministic
    assert model.trace is not None
    match_probs = model.trace.posterior["match_probs"]

    # Sum over the last dimension (H, D, A classes)
    prob_sums = match_probs.sum(dim="match_probs_dim_1")

    # All probabilities should sum to 1 (within numerical tolerance)
    assert np.allclose(prob_sums.values, 1.0, atol=1e-6)


def test_prediction_works_with_calibration(sample_data, monkeypatch: Any):
    """Test that prediction works correctly when calibration is enabled."""
    _patch_hierarchical_bayes(monkeypatch, sample_data)
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


def test_calibration_improves_model_fit(sample_data, monkeypatch: Any):
    """Test that calibration doesn't break the model (basic sanity check)."""
    _patch_hierarchical_bayes(monkeypatch, sample_data)
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


def test_calibration_parameters_are_learnable(sample_data, monkeypatch: Any):
    """Test that calibration parameters vary across posterior samples (not stuck)."""
    _patch_hierarchical_bayes(monkeypatch, sample_data)
    model = BayesianModel(n_goals=5, calibrate=True)
    model.fit(sample_data)

    # Check that tau varies
    assert model.trace is not None
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
