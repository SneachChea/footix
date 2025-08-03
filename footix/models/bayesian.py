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

    def __init__(self, n_teams: int, n_goals: int):
        self.n_teams = n_teams
        self.n_goals = n_goals
        self.trace: az.InferenceData | None = None
        self.label = preprocessing.LabelEncoder()

    @verify_required_column(column_names={"home_team", "away_team", "fthg", "ftag"})
    def fit(self, X_train: pd.DataFrame):
        x_train_cop = X_train.copy(deep=False)
        teams = pd.concat([X_train["home_team"], X_train["away_team"]]).unique()
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
            "alpha": p["alpha_NB"].mean(("chain", "draw")).values.item(),
            "rho": p["rho_diag"].mean(("chain", "draw")).values.item(),
        }
        return self._cached_means

    def predict(
        self, home_team: str, away_team: str, join_distribution: bool = False
    ) -> GoalMatrix:
        home_id, away_id = self.label.transform([home_team, away_team])

        means = self._posterior_means()
        home_mu = np.exp(
            means["intercept"]
            + means["home"][home_id]
            + means["atts"][home_id]
            + means["defs"][away_id]
        )
        away_mu = np.exp(means["intercept"] + means["atts"][away_id] + means["defs"][home_id])

        ks = np.arange(self.n_goals)
        alpha = means["alpha"]
        home_pmf = stats.nbinom.pmf(ks, alpha, alpha / (alpha + home_mu))
        away_pmf = stats.nbinom.pmf(ks, alpha, alpha / (alpha + away_mu))
        if join_distribution:
            rho = means.get("rho_diag", 0.0)  # fallback if not present
            corr = np.ones((self.n_goals, self.n_goals))
            np.fill_diagonal(corr, 1.0 + rho)
            return GoalMatrix(home_pmf, away_pmf, correlation_matrix=corr)

        return GoalMatrix(home_pmf, away_pmf)

    def goal_expectation(self, home_team_id: int, away_team_id: int):
        posterior = self.trace.posterior  # type:ignore

        # posterior means
        home = posterior["home"].mean(dim=["chain", "draw"]).values
        intercept = posterior["intercept"].mean(dim=["chain", "draw"]).values
        atts = posterior["atts"].mean(dim=["chain", "draw"]).values
        defs = posterior["defs"].mean(dim=["chain", "draw"]).values
        alpha = posterior["alpha_NB"].mean(dim=["chain", "draw"]).values

        # linear predictors → expected counts
        home_mu = np.exp(intercept + home[home_team_id] + atts[home_team_id] + defs[away_team_id])
        away_mu = np.exp(intercept + atts[away_team_id] + defs[home_team_id])

        # return both expectations and the dispersion α
        return home_mu, away_mu, alpha

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

        team_id = self.label.transform([home_team, away_team])

        home_team_id = team_id[0]
        away_team_id = team_id[1]
        _posterior = self.trace.posterior  # type:ignore
        home = _posterior["home"].stack(sample=("chain", "draw")).values
        atts = _posterior["atts"].stack(sample=("chain", "draw")).values
        defs = _posterior["defs"].stack(sample=("chain", "draw")).values
        intercept = _posterior["intercept"].stack(sample=("chain", "draw")).values
        alpha = _posterior["alpha_NB"].stack(sample=("chain", "draw")).values
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
            alpha_i = alpha[i]

            home_goals = np.random.negative_binomial(alpha_i, alpha_i / (alpha_i + mu_home), 150)
            away_goals = np.random.negative_binomial(alpha_i, alpha_i / (alpha_i + mu_away), 150)
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
        with pm.Model():
            # Use pm.Data for the observed data and covariates
            goals_home_data = pm.Data("goals_home", goals_home_obs)
            goals_away_data = pm.Data("goals_away", goals_away_obs)
            home_team_data = pm.Data("home_team", home_team)
            away_team_data = pm.Data("away_team", away_team)

            # Home advantage and intercept
            home = pm.Normal("home", mu=0, sigma=1, shape=self.n_teams)
            intercept = pm.Normal("intercept", mu=0.4, sigma=0.5)

            rho_diag = pm.HalfNormal("rho_diag", sigma=0.3)

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
            alpha = pm.HalfCauchy("alpha_NB", 2.0)  # dispersion (α → 0 recovers Poisson)

            # Calculate theta for home and away
            home_theta = pm.math.exp(
                intercept + home[home_team_data] + atts[home_team_data] + defs[away_team_data]
            )
            away_theta = pm.math.exp(intercept + atts[away_team_data] + defs[home_team_data])
            pm.NegativeBinomial(
                "home_goals",
                mu=home_theta,
                alpha=alpha,  # NB parameterisation: (μ, α)
                observed=goals_home_data,
            )
            pm.NegativeBinomial(
                "away_goals",
                mu=away_theta,
                alpha=alpha,
                observed=goals_away_data,
            )
            # ─── inflation de la diagonale (toutes valeurs de buts) ──
            same_score = pt.eq(goals_home_obs, goals_away_obs)
            log_weight = pt.switch(same_score, pt.log1p(rho_diag), 0.0)  # log(1+ρ)
            pm.Potential("draw_inflation", log_weight)

            # Goal likelihood
            # pm.Poisson("home_goals", mu=home_theta, observed=goals_home_data)
            # pm.Poisson("away_goals", mu=away_theta, observed=goals_away_data)
            # Sample with improved settings
            trace = pm.sample(
                2000,
                tune=1000,
                cores=os.cpu_count(),
                target_accept=0.95,
                return_inferencedata=True,
                nuts_sampler="numpyro",
                init="adapt_diag_grad",
            )
        return trace
