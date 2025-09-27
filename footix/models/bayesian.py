from __future__ import annotations

import os
import warnings
from functools import cache
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import scipy.stats as stats
from sklearn import preprocessing

from footix.models.score_matrix import GoalMatrix
from footix.utils.decorators import verify_required_column
from footix.utils.typing import SampleProbaResult


class BayesianModel:
    """Bayesian hierarchical model for football scores using a Negative Binomial likelihood.

    Attributes
    ----------
    n_teams : int
        Number of distinct teams in the league.
    n_goals : int
        Maximum number of goals considered when computing score probabilities.
    trace : arviz.InferenceData | None
        Posterior samples after calling `fit`. None until the model is fitted.

    """

    def __init__(self, n_goals: int, n_teams: int | None = None, calibrate: bool = False):
        self.n_teams = n_teams
        self.n_goals = n_goals
        self.calibrate = calibrate
        self.trace: az.InferenceData | None = None
        self.label = preprocessing.LabelEncoder()

    @verify_required_column(column_names={"home_team", "away_team", "fthg", "ftag"})
    def fit(self, X_train: pd.DataFrame):
        x_train_cop = X_train.copy(deep=False)
        teams = pd.concat([X_train["home_team"], X_train["away_team"]]).unique()
        if self.n_teams is None:
            self.n_teams = len(teams)
        elif self.n_teams != len(teams):
            raise ValueError(
                f"Teams in training data do not match the initialized teams. "
                f"Expected: {self.n_teams}, got: {teams}."
            )

        self.label.fit(teams)  # type: ignore
        x_train_cop["home_team_id"] = self.label.transform(X_train["home_team"])
        x_train_cop["away_team_id"] = self.label.transform(X_train["away_team"])

        goals_home_obs = x_train_cop["fthg"].to_numpy()
        goals_away_obs = x_train_cop["ftag"].to_numpy()
        home_team = x_train_cop["home_team_id"].to_numpy()
        away_team = x_train_cop["away_team_id"].to_numpy()
        self.trace = self.hierarchical_bayes(goals_home_obs, goals_away_obs, home_team, away_team)

    @cache
    def _posterior_means(self) -> dict[str, np.ndarray]:
        p = self.trace.posterior  # type:ignore
        self._cached_means = {
            "home": p["home"].mean(("chain", "draw")).values,
            "intercept": p["intercept"].mean(("chain", "draw")).values.item(),
            "atts": p["atts"].mean(("chain", "draw")).values,
            "defs": p["defs"].mean(("chain", "draw")).values,
        }
        return self._cached_means

    def predict(self, home_team: str, away_team: str) -> GoalMatrix:
        home_id, away_id = self.label.transform([home_team, away_team])

        home_mu, away_mu = self.goal_expectation(home_team_id=home_id, away_team_id=away_id)

        ks = np.arange(self.n_goals)
        home_pmf = stats.poisson.pmf(ks, home_mu)
        away_pmf = stats.poisson.pmf(ks, away_mu)
        return GoalMatrix(home_pmf, away_pmf)

    def goal_expectation(self, home_team_id: int, away_team_id: int):
        posterior = self.trace.posterior  # type:ignore

        # posterior means
        home = posterior["home"].mean(dim=["chain", "draw"]).values
        intercept = posterior["intercept"].mean(dim=["chain", "draw"]).values
        atts = posterior["atts"].mean(dim=["chain", "draw"]).values
        defs = posterior["defs"].mean(dim=["chain", "draw"]).values
        # linear predictors → expected counts
        home_mu = np.exp(intercept + home[home_team_id] + atts[home_team_id] + defs[away_team_id])
        away_mu = np.exp(intercept + atts[away_team_id] + defs[home_team_id])

        # return both expectations and the dispersion α
        return home_mu, away_mu

    @cache
    def get_samples(self, home_team: str, away_team: str, **kwargs: Any) -> SampleProbaResult:
        """Generates posterior predictive samples for the specified home and away teams based on
        the model.

            home_team (str): The name of the home team.
            away_team (str): The name of the away team.

            tuple[np.ndarray, np.ndarray]:
                A tuple containing two one-dimensional numpy arrays:
                    - The first array represents the sampled lambda values for the home team.
                    - The second array represents the sampled lambda values for the away team.
        Notes:
            This function transforms the team names into their corresponding indices, retrieves
            the posterior samples for model parameters from the trace, computes the expected
            goal rates (lambda values) for both teams, and flattens the arrays to provide a
            simplified output.

        """
        if kwargs:
            warnings.warn(
                f"Ignoring unexpected keyword arguments: {list(kwargs.keys())}", stacklevel=2
            )

        home_team_id, away_team_id = self.label.transform([home_team, away_team])

        _posterior = self.trace.posterior  # type:ignore
        home = _posterior["home"].stack(sample=("chain", "draw")).values
        atts = _posterior["atts"].stack(sample=("chain", "draw")).values
        defs = _posterior["defs"].stack(sample=("chain", "draw")).values
        intercept = _posterior["intercept"].stack(sample=("chain", "draw")).values
        n_samples = intercept.shape[0]

        prob_H_list = []
        prob_D_list = []
        prob_A_list = []

        for i in range(n_samples):
            mu_home = np.exp(
                intercept[i]
                + home[home_team_id, i]
                + atts[home_team_id, i]
                + defs[away_team_id, i]
            )
            mu_away = np.exp(intercept[i] + atts[away_team_id, i] + defs[home_team_id, i])

            home_goals = np.random.poisson(mu_home, 150)
            away_goals = np.random.poisson(mu_away, 150)
            prob_H = np.mean(home_goals > away_goals)
            prob_D = np.mean(home_goals == away_goals)
            prob_A = np.mean(home_goals < away_goals)

            prob_H_list.append(prob_H)
            prob_D_list.append(prob_D)
            prob_A_list.append(prob_A)

        return SampleProbaResult(
            proba_home=np.asarray(prob_H_list),
            proba_draw=np.asarray(prob_D_list),
            proba_away=np.asarray(prob_A_list),
        )

    def hierarchical_bayes(
        self,
        goals_home_obs: np.ndarray,
        goals_away_obs: np.ndarray,
        home_team: np.ndarray,
        away_team: np.ndarray,
    ) -> az.InferenceData:
        match_obs = np.where(
            goals_home_obs > goals_away_obs, 0, np.where(goals_home_obs == goals_away_obs, 1, 2)
        )
        with pm.Model():
            # Use pm.Data for the observed data and covariates
            goals_home_data = pm.Data("goals_home", goals_home_obs)
            goals_away_data = pm.Data("goals_away", goals_away_obs)
            home_team_data = pm.Data("home_team", home_team)
            away_team_data = pm.Data("away_team", away_team)

            # Home advantage and intercept
            home = pm.Normal("home", mu=0, sigma=1, shape=self.n_teams)
            intercept = pm.Normal("intercept", mu=0.4, sigma=0.5)

            # Attack ratings with non-centered parameterization
            tau_att = pm.HalfNormal("tau_att", sigma=2)
            raw_atts = pm.Normal("raw_atts", mu=0, sigma=1, shape=self.n_teams)
            atts_uncentered = raw_atts * tau_att
            atts = pm.Deterministic("atts", atts_uncentered - pm.math.mean(atts_uncentered))
            # Defence ratings with non-centered parameterization
            tau_def = pm.HalfNormal("tau_def", sigma=2)
            raw_defs = pm.Normal("raw_defs", mu=0, sigma=1, shape=self.n_teams)
            defs_uncentered = raw_defs * tau_def
            defs = pm.Deterministic("defs", defs_uncentered - pm.math.mean(defs_uncentered))

            # Calculate theta for home and away
            home_theta = pm.math.exp(
                intercept + home[home_team_data] + atts[home_team_data] + defs[away_team_data]
            )
            away_theta = pm.math.exp(intercept + atts[away_team_data] + defs[home_team_data])
            pm.Poisson(
                "home_goals",
                mu=home_theta,
                observed=goals_home_data,
            )
            pm.Poisson(
                "away_goals",
                mu=away_theta,
                observed=goals_away_data,
            )

            if self.calibrate:
                match_res_data = pm.Data("match_res", match_obs)

                eps = 1e-12  # tiny floor

                p_H = pm.Deterministic(
                    "p_H", var=p_skellam_gt0_continuity(mu1=home_theta, mu2=away_theta)
                )
                p_D = pm.Deterministic("p_D", var=p_skellam_eq0(mu1=home_theta, mu2=away_theta))
                p_A = pm.Deterministic("p_A", var=1 - p_H - p_D)

                match_probs_raw = pt.stack([p_H, p_D, p_A], axis=1)
                match_probs_raw = pt.clip(match_probs_raw, eps, 1.0)  # strictly > 0
                match_probs_raw = match_probs_raw / match_probs_raw.sum(axis=1, keepdims=True)
                match_probs = pm.Deterministic("match_probs", match_probs_raw)
                pm.Categorical("match_outcomes", p=match_probs, observed=match_res_data)
                # Goal likelihood
            # pm.Poisson("home_goals", mu=home_theta, observed=goals_home_data)
            # pm.Poisson("away_goals", mu=away_theta, observed=goals_away_data)
            # Sample with improved settings
            trace = pm.sample(
                2000,
                tune=1000,
                cores=min(4, os.cpu_count() or 1),
                target_accept=0.95,
                return_inferencedata=True,
                nuts_sampler="numpyro",
                init="adapt_diag_grad",
            )
        return trace


def p_skellam_gt0_continuity(mu1, mu2, eps=1e-9):
    """Approx P(K>0) for K ~ Skellam(mu1, mu2) using normal approx + continuity correction.

    Returns a PyTensor node usable inside a PyMC model.

    """
    var = pm.math.maximum(mu1 + mu2, eps)  # stabilité num.
    z = (mu1 - mu2 - 0.5) / pm.math.sqrt(var)  # correction de continuité
    # Φ(z) : CDF normale standard
    return (1.0 + pm.math.erf(z / pm.math.sqrt(2.0))) / 2.0


def p_skellam_eq0(mu1, mu2, eps=1e-9):
    """Probabilité exacte P(K = 0) pour K ~ Skellam(mu1, mu2) =  exp(-(mu1+mu2)) * I0(
    2*sqrt(mu1*mu2) )

    - mu1, mu2 peuvent être scalaires ou tenseurs aléatoires (stochastiques du modèle)
    - eps évite les problèmes de dérivées pour mu1*mu2 = 0

    """
    x = 2.0 * pm.math.sqrt(pm.math.maximum(mu1 * mu2, eps))
    return pm.math.exp(-(mu1 + mu2)) * pt.i0(x)  # pt.i0 : Bessel I0
