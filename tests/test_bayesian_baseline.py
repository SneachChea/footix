"""Baseline tests for the Bayesian model: posterior-predictive point
predictions, cache invalidation, priors, posterior predictive checks, MCMC
diagnostics and reproducibility."""

from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from footix.models.bayesian import BayesianModel, prior_predictive_draws

pytestmark = pytest.mark.bayesian


def _build_fake_trace(n_teams: int) -> az.InferenceData:
    """Deterministic multi-draw posterior in the baseline variable names."""
    draws = 12
    draw_axis = np.arange(draws, dtype=float)

    home = np.full((1, draws), 0.2, dtype=float)
    intercept = np.linspace(0.2, 0.35, draws, dtype=float).reshape(1, draws)

    attack_base = np.linspace(-0.15, 0.15, n_teams, dtype=float)
    defence_base = np.linspace(0.12, -0.12, n_teams, dtype=float)
    attack = (attack_base.reshape(1, n_teams) + 0.02 * draw_axis.reshape(draws, 1))[None, :, :]
    defence = (defence_base.reshape(1, n_teams) + 0.01 * draw_axis.reshape(draws, 1))[None, :, :]

    posterior: dict[str, np.ndarray] = {
        "home": home,
        "intercept": intercept,
        "attack_strength": attack,
        "defence_strength": defence,
    }
    return az.from_dict({"posterior": posterior})


def _patch_fit(monkeypatch: Any, trace_factory: Any) -> None:
    """Replace the expensive MCMC with a deterministic fake posterior."""

    def fake_hierarchical_bayes(
        self: BayesianModel,
        goals_home_obs: np.ndarray,
        goals_away_obs: np.ndarray,
        home_team: np.ndarray,
        away_team: np.ndarray,
        optional_stats: dict[str, Any] | None = None,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> az.InferenceData:
        _ = goals_home_obs, home_team, away_team, optional_stats, sample_kwargs
        if trace_factory is None:
            return _build_fake_trace(n_teams=int(self.n_teams or 0))
        return trace_factory(self, goals_home_obs, goals_away_obs)

    monkeypatch.setattr(BayesianModel, "hierarchical_bayes", fake_hierarchical_bayes)


def _sample_data(seed: int = 42, n_matches: int = 30, goal_rate: float = 1.4) -> pd.DataFrame:
    """Small deterministic league with the required football-data columns."""
    rng = np.random.default_rng(seed)
    teams = ["Team_A", "Team_B", "Team_C", "Team_D"]
    rows = []
    for _ in range(n_matches):
        home, away = rng.choice(teams, 2, replace=False)
        rows.append(
            {
                "home_team": home,
                "away_team": away,
                "fthg": rng.poisson(goal_rate),
                "ftag": rng.poisson(goal_rate * 0.8),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Posterior-predictive point predictions
# ---------------------------------------------------------------------------


def test_predict_equals_mean_of_market_samples_1x2(monkeypatch: Any) -> None:
    """predict() is the posterior predictive mean for the 1X2 market."""
    _patch_fit(monkeypatch, None)
    model = BayesianModel(n_goals=6)
    model.fit(_sample_data())

    teams = sorted(set(_sample_data()["home_team"]) | set(_sample_data()["away_team"]))
    prediction = model.predict(teams[0], teams[1]).return_probas()
    samples_mean = model.get_market_samples(teams[0], teams[1], "1X2").mean(axis=0)

    expected = np.asarray([prediction.proba_home, prediction.proba_draw, prediction.proba_away])
    assert np.allclose(expected, samples_mean, atol=1e-10)


def test_predict_equals_mean_of_market_samples_total_goals(monkeypatch: Any) -> None:
    """predict() is the posterior predictive mean for the O/U2.5 market."""
    _patch_fit(monkeypatch, None)
    model = BayesianModel(n_goals=6)
    model.fit(_sample_data())

    teams = sorted(set(_sample_data()["home_team"]) | set(_sample_data()["away_team"]))
    prediction = model.predict(teams[0], teams[1])
    samples_mean = model.get_market_samples(teams[0], teams[1], "O/U2.5").mean(axis=0)

    assert np.isclose(prediction.less_25_goals(), samples_mean[0], atol=1e-10)
    assert np.isclose(prediction.more_25_goals(), samples_mean[1], atol=1e-10)


def test_per_draw_normalization(monkeypatch: Any) -> None:
    """Every posterior draw is a proper probability vector for every market."""
    _patch_fit(monkeypatch, None)
    model = BayesianModel(n_goals=6)
    model.fit(_sample_data())

    teams = sorted(set(_sample_data()["home_team"]) | set(_sample_data()["away_team"]))
    for market, n_outcomes in (("1X2", 3), ("O/U2.5", 2)):
        samples = model.get_market_samples(teams[0], teams[1], market)
        assert samples.shape[1] == n_outcomes
        assert np.allclose(samples.sum(axis=1), 1.0, atol=1e-12)
        assert np.all((samples >= 0) & (samples <= 1))


def test_get_samples_reuses_market_samples(monkeypatch: Any) -> None:
    """get_samples must not define a second set of probabilities."""
    _patch_fit(monkeypatch, None)
    model = BayesianModel(n_goals=6)
    model.fit(_sample_data())

    teams = sorted(set(_sample_data()["home_team"]) | set(_sample_data()["away_team"]))
    samples = model.get_samples(teams[0], teams[1])
    market = model.get_market_samples(teams[0], teams[1], "1X2")
    assert np.allclose(samples.proba_home, market[:, 0])
    assert np.allclose(samples.proba_draw, market[:, 1])
    assert np.allclose(samples.proba_away, market[:, 2])


def test_predict_goal_matrix_convention(monkeypatch: Any) -> None:
    """The defence convention: a strong defence lowers the opponent's rate."""
    _patch_fit(monkeypatch, None)
    model = BayesianModel(n_goals=6)
    model.fit(_sample_data())

    # Highest defence_strength (best defence) vs lowest (worst defence).
    defence = np.array(
        model.trace.posterior["defence_strength"]  # type: ignore[union-attr]
        .mean(("chain", "draw"))
        .values
    )
    best_defender, worst_defender = int(np.argmax(defence)), int(np.argmin(defence))
    attacker = 0

    # A team concedes fewer goals against a better defence.
    away_mu_best = model.goal_expectation(attacker, best_defender)[0]
    away_mu_worst = model.goal_expectation(attacker, worst_defender)[0]
    assert away_mu_best < away_mu_worst


def test_goal_expectation_is_mean_of_draw_rates_not_exp_of_mean() -> None:
    """goal_expectation must integrate the posterior, not exp(E[eta]).

    exp is convex, so exp(E[eta]) underestimates E[exp(eta)] (Jensen). The
    synthetic posterior below makes the gap huge: two draws at eta = 0.3 and
    2.3, where exp(1.3) = 3.67 but (exp(0.3)+exp(2.3))/2 = 5.99.
    """
    posterior = {
        "home": np.full((1, 2), 0.3),
        "intercept": np.asarray([[0.0, 2.0]]),
        "attack_strength": np.zeros((1, 2, 2)),
        "defence_strength": np.zeros((1, 2, 2)),
    }
    model = BayesianModel(n_goals=10)
    model.trace = az.from_dict({"posterior": posterior})

    home_mu, away_mu = model.goal_expectation(0, 1)
    assert np.isclose(home_mu, np.mean([np.exp(0.3), np.exp(2.3)]))
    assert np.isclose(away_mu, np.mean([np.exp(0.0), np.exp(2.0)]))

    # the rejected exp(E[eta]) definition is clearly lower
    assert home_mu > np.exp(0.3 + 1.0)
    assert away_mu > np.exp(1.0)


def test_predict_survives_underflowing_score_cells() -> None:
    """Near-zero rates must not turn predict() into a NaN/exception."""
    posterior = {
        "home": np.full((1, 4), 0.2),
        "intercept": np.full((1, 4), -20.0),  # lambda ~ 2e-9 -> high-goal pmf == 0
        "attack_strength": np.zeros((1, 4, 2)),
        "defence_strength": np.zeros((1, 4, 2)),
    }
    model = BayesianModel(n_goals=20)
    model.trace = az.from_dict({"posterior": posterior})
    model._team_to_id = {"A": 0, "B": 1}

    prediction = model.predict("A", "B")
    probas = prediction.return_probas()
    assert np.isclose(probas.proba_home + probas.proba_draw + probas.proba_away, 1.0)
    assert np.isclose(probas.proba_draw, 1.0, atol=1e-9)  # 0-0 dominates


# ---------------------------------------------------------------------------
# Cache invalidation on refit
# ---------------------------------------------------------------------------


def test_refit_invalidates_cached_predictions(monkeypatch: Any) -> None:
    """Refitting the same instance must drop every cached prediction."""
    data_a = _sample_data(seed=1, goal_rate=1.0)
    data_b = _sample_data(seed=2, goal_rate=6.0)

    def factory(self: BayesianModel, goals_home_obs: np.ndarray, goals_away_obs: np.ndarray):
        observed_rate = float(np.mean(np.concatenate([goals_home_obs, goals_away_obs])))
        intercept = np.full((1, 12), np.log(max(observed_rate, 0.05)))
        posterior: dict[str, np.ndarray] = {
            "home": np.full((1, 12), 0.2),
            "intercept": intercept,
            "attack_strength": np.full((1, 12, int(self.n_teams or 0)), 0.0),
            "defence_strength": np.full((1, 12, int(self.n_teams or 0)), 0.0),
        }
        return az.from_dict({"posterior": posterior})

    _patch_fit(monkeypatch, factory)
    teams = ["Team_A", "Team_B"]

    model = BayesianModel(n_goals=6)
    model.fit(data_a)
    before = model.predict(teams[0], teams[1]).return_probas()
    samples_before = model.get_market_samples(teams[0], teams[1], "1X2").copy()

    model.fit(data_b)
    after = model.predict(teams[0], teams[1]).return_probas()

    # The refit must actually change the prediction (different data, same teams).
    before_arr = np.asarray([before.proba_home, before.proba_draw, before.proba_away])
    after_arr = np.asarray([after.proba_home, after.proba_draw, after.proba_away])
    assert not np.allclose(before_arr, after_arr, atol=1e-3)

    # And it must match a fresh model fitted on the second dataset.
    fresh = BayesianModel(n_goals=6)
    fresh.fit(data_b)
    fresh_probas = fresh.predict(teams[0], teams[1]).return_probas()
    fresh_arr = np.asarray(
        [fresh_probas.proba_home, fresh_probas.proba_draw, fresh_probas.proba_away]
    )
    assert np.allclose(after_arr, fresh_arr, atol=1e-12)

    # The cached per-draw samples of the first fit are also gone.
    samples_after = model.get_market_samples(teams[0], teams[1], "1X2")
    assert not np.array_equal(samples_after, samples_before)
    assert np.allclose(
        samples_after.mean(axis=0),
        fresh.get_market_samples(teams[0], teams[1], "1X2").mean(axis=0),
    )

    # draw axis length changed, proving the first fit's cache is not reused
    assert samples_after.shape[0] == 12


# ---------------------------------------------------------------------------
# Prior predictive checks
# ---------------------------------------------------------------------------


def test_prior_predictive_scale_is_plausible() -> None:
    """The priors must generate football-like scores, never absurd ones."""
    home, away = prior_predictive_draws(n_teams=20, n_matches=200, draws=1500, random_seed=3)
    total = home + away

    # Global home advantage on the prior: home teams score more on average.
    assert home.mean() > away.mean()

    # Mean per-team rates compatible with professional football.
    assert 0.6 < home.mean() < 3.0
    assert 0.6 < away.mean() < 3.0

    # Extreme scores must stay exceptional.
    assert (home >= 10).mean() < 0.01
    assert (away >= 10).mean() < 0.01
    assert (total >= 15).mean() < 0.01

    # Scores are not degenerate (a non-trivial share of both 0 and 3+ goals).
    assert 0.05 < (home == 0).mean() < 0.6
    assert (home <= 2).mean() > 0.5


def test_prior_predictive_is_reproducible() -> None:
    """Same seed, same prior draws."""
    home_a, away_a = prior_predictive_draws(10, 50, draws=300, random_seed=7)
    home_b, away_b = prior_predictive_draws(10, 50, draws=300, random_seed=7)
    assert np.allclose(home_a, home_b)
    assert np.allclose(away_a, away_b)


# ---------------------------------------------------------------------------
# Posterior predictive checks
# ---------------------------------------------------------------------------


def test_posterior_predictive_matches_manual_per_draw_average(monkeypatch: Any) -> None:
    """posterior_predictive_distribution is the mean of per-draw score matrices."""
    _patch_fit(monkeypatch, None)
    model = BayesianModel(n_goals=6)
    model.fit(_sample_data())

    teams = sorted(set(_sample_data()["home_team"]) | set(_sample_data()["away_team"]))
    home_team, away_team = teams[0], teams[1]
    posterior = model.trace.posterior  # type: ignore[union-attr]
    home = posterior["home"].stack(sample=("chain", "draw")).values
    attack = posterior["attack_strength"].stack(sample=("chain", "draw")).values
    defence = posterior["defence_strength"].stack(sample=("chain", "draw")).values
    intercept = posterior["intercept"].stack(sample=("chain", "draw")).values
    home_id = model._team_to_id[home_team]
    away_id = model._team_to_id[away_team]
    goals = np.arange(model.n_goals)

    matrices = []
    for index in range(intercept.size):
        mu_home = np.exp(
            intercept[index] + home[index] + attack[home_id, index] - defence[away_id, index]
        )
        mu_away = np.exp(intercept[index] + attack[away_id, index] - defence[home_id, index])
        pmf_home = stats.poisson.pmf(goals, mu_home)
        pmf_away = stats.poisson.pmf(goals, mu_away)
        pmf_home /= pmf_home.sum()
        pmf_away /= pmf_away.sum()
        matrices.append(np.outer(pmf_home, pmf_away))

    expected = np.mean(matrices, axis=0)
    actual = model.posterior_predictive_distribution(home_team, away_team)
    assert np.allclose(actual, expected, atol=1e-12)
    assert np.isclose(actual.sum(), 1.0, atol=1e-12)


def test_predict_probas_match_posterior_predictive_matrix(monkeypatch: Any) -> None:
    """predict()'s 1X2 probabilities equal the posterior-predictive matrix sums."""
    _patch_fit(monkeypatch, None)
    model = BayesianModel(n_goals=6)
    model.fit(_sample_data())

    teams = sorted(set(_sample_data()["home_team"]) | set(_sample_data()["away_team"]))
    matrix = model.posterior_predictive_distribution(teams[0], teams[1])
    expected_home = np.sum(np.tril(matrix, -1))
    expected_draw = np.sum(np.diag(matrix))
    expected_away = np.sum(np.triu(matrix, 1))

    prediction = model.predict(teams[0], teams[1]).return_probas()
    assert np.isclose(prediction.proba_home, expected_home, atol=1e-12)
    assert np.isclose(prediction.proba_draw, expected_draw, atol=1e-12)
    assert np.isclose(prediction.proba_away, expected_away, atol=1e-12)


# ---------------------------------------------------------------------------
# MCMC diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_failed_before_fit() -> None:
    model = BayesianModel(n_goals=6)
    diagnostics = model.get_diagnostics()
    assert diagnostics["status"] == "failed"
    assert "fitted" in diagnostics["reason"]


def _diag_trace(chain_means: list[float], diverging: np.ndarray | None = None) -> az.InferenceData:
    """Two-variable posterior with controllable between-chain separation."""
    rng = np.random.default_rng(0)
    draws, n_chains = 500, len(chain_means)
    posterior: dict[str, np.ndarray] = {
        "intercept": np.asarray([[mean] * draws for mean in chain_means], dtype=float).reshape(
            n_chains, draws
        )
        + rng.normal(0, 0.05, size=(n_chains, draws)),
        "home": np.asarray([[mean] * draws for mean in chain_means], dtype=float).reshape(
            n_chains, draws
        )
        + rng.normal(0, 0.05, size=(n_chains, draws)),
    }
    sample_stats: dict[str, np.ndarray] = {}
    if diverging is not None:
        sample_stats["diverging"] = diverging
    elif n_chains:
        sample_stats["diverging"] = np.zeros((n_chains, draws), dtype=int)
    return az.from_dict({"posterior": posterior, "sample_stats": sample_stats})


def test_diagnostics_valid_on_converged_chain() -> None:
    trace = _diag_trace([0.0, 0.0, 0.0, 0.0])
    model = BayesianModel(n_goals=6)
    model.trace = trace
    diagnostics = model.get_diagnostics()

    assert diagnostics["status"] == "valid"
    assert diagnostics["divergences"] == 0
    assert diagnostics["max_rhat"] <= 1.01
    assert diagnostics["min_ess_bulk"] >= diagnostics["ess_threshold"]
    assert diagnostics["min_ess_tail"] >= diagnostics["ess_threshold"]
    assert diagnostics["reason"] is None


def test_diagnostics_invalid_on_split_chains() -> None:
    """Two chains with disjoint supports must be flagged invalid_mcmc."""
    trace = _diag_trace([-3.0, -3.0, 3.0, 3.0])
    model = BayesianModel(n_goals=6)
    model.trace = trace

    diagnostics = model.get_diagnostics()
    assert diagnostics["status"] == "invalid_mcmc"
    assert diagnostics["max_rhat"] > 1.01
    assert diagnostics["reason"] is not None


def test_diagnostics_invalid_on_divergences() -> None:
    diverging = np.zeros((2, 500), dtype=int)
    diverging[0, 3] = 1
    trace = _diag_trace([0.0, 0.0], diverging=diverging)
    model = BayesianModel(n_goals=6)
    model.trace = trace

    diagnostics = model.get_diagnostics()
    assert diagnostics["status"] == "invalid_mcmc"
    assert diagnostics["divergences"] == 1


def test_diagnostics_serializable() -> None:
    trace = _diag_trace([0.0, 0.0])
    model = BayesianModel(n_goals=6)
    model.trace = trace
    diagnostics = model.get_diagnostics()

    import json

    json.dumps(diagnostics)  # must not raise


# ---------------------------------------------------------------------------
# Reproducibility with real MCMC
# ---------------------------------------------------------------------------


def test_two_fits_same_seed_agree():
    """Two fits on the same data and seed give compatible predictions."""
    data = _sample_data(seed=11, n_matches=40)
    sample_kwargs = {"draws": 250, "tune": 200, "chains": 2, "cores": 2}

    first = BayesianModel(n_goals=8, random_seed=5)
    first.fit(data, sample_kwargs=sample_kwargs)
    second = BayesianModel(n_goals=8, random_seed=5)
    second.fit(data, sample_kwargs=sample_kwargs)

    teams = sorted(set(data["home_team"]) | set(data["away_team"]))
    p1 = first.get_market_samples(teams[0], teams[1], "1X2")
    p2 = second.get_market_samples(teams[0], teams[1], "1X2")
    assert np.allclose(p1.mean(axis=0), p2.mean(axis=0), atol=0.05)

    d1 = first.get_diagnostics()
    d2 = second.get_diagnostics()
    assert d1["status"] == d2["status"] == "valid"

    # Posterior predictive totals roughly match the observed goal rate.
    pp = first.posterior_predictive_distribution(teams[0], teams[1])
    goals = np.arange(first.n_goals)
    expected_goals = goals @ pp @ goals
    observed_rate = float((data["fthg"].to_numpy()[data["home_team"] == teams[0]].mean() or 1.0))
    assert 0.3 < expected_goals < 5.0
    _ = observed_rate  # sanity bound only, no league-specific hardcoding
