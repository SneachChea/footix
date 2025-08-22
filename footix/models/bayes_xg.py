import os
import warnings
from copy import copy
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import scipy.stats as stats
from sklearn import preprocessing

from footix.models.score_matrix import GoalMatrix
from footix.utils.decorators import verify_required_column
from footix.utils.typing import SampleProbaResult


class XGBayesian:
    def __init__(self, n_teams: int, n_goals: int):
        self.n_teams = n_teams
        self.n_goals = n_goals
        self.label = preprocessing.LabelEncoder()

    @verify_required_column(
        column_names={"home_team", "away_team", "fthg", "ftag", "fthxg", "ftaxg"}
    )
    def fit(self, X_train: pd.DataFrame):
        x_train_cop = copy(X_train)
        self.label.fit(X_train["home_team"])  # type: ignore
        x_train_cop["home_team_id"] = self.label.transform(X_train["home_team"])
        x_train_cop["away_team_id"] = self.label.transform(X_train["away_team"])

        goals_home_obs = x_train_cop["fthg"].to_numpy()
        goals_away_obs = x_train_cop["ftag"].to_numpy()
        xg_home_obs = np.clip(x_train_cop["fthxg"].to_numpy(), a_min=1e-5, a_max=None)
        xg_away_obs = np.clip(x_train_cop["ftaxg"].to_numpy(), a_min=1e-5, a_max=None)
        home_team = x_train_cop["home_team_id"].to_numpy()
        away_team = x_train_cop["away_team_id"].to_numpy()
        self.trace = self.hierarchical_joint_xg_goals(
            goals_home_obs, goals_away_obs, xg_home_obs, xg_away_obs, home_team, away_team
        )

    def predict(self, home_team: str, away_team: str, **kwargs: Any) -> GoalMatrix:
        if kwargs:
            warnings.warn(
                f"Ignoring unexpected keyword arguments: {list(kwargs.keys())}", stacklevel=2
            )
        team_id = self.label.transform([home_team, away_team])

        home_goal_expectation, away_goal_expectation = self.goal_expectation(
            home_team_id=team_id[0], away_team_id=team_id[1]
        )

        home_probs = stats.poisson.pmf(range(self.n_goals), home_goal_expectation)
        away_probs = stats.poisson.pmf(range(self.n_goals), away_goal_expectation)

        goals_matrix = GoalMatrix(home_probs, away_probs)
        return goals_matrix

    def goal_expectation(self, home_team_id: int, away_team_id: int):
        post = self.trace.posterior

        alpha = post["alpha_xg"].mean(("chain", "draw")).item()
        home_adv = post["home_adv_xg"].mean(("chain", "draw")).item()
        att = post["att"].mean(("chain", "draw")).values
        deff = post["def"].mean(("chain", "draw")).values
        alpha_g = post["alpha_goal"].mean(("chain", "draw")).item()
        delta = post["delta"].mean(("chain", "draw")).item()

        eta_h = alpha + home_adv + att[home_team_id] - deff[away_team_id]
        eta_a = alpha + att[away_team_id] - deff[home_team_id]

        home_theta = np.exp(alpha_g + delta * eta_h)
        away_theta = np.exp(alpha_g + delta * eta_a)
        return home_theta, away_theta

    def get_samples(self, home_team: str, away_team: str, **kwargs: Any) -> SampleProbaResult:
        """Posterior‑predictive λ samples for one fixture, including Gamma‑sampled latent‑xG
        uncertainty.

        Parameters
        ----------
        home_team, away_team : str
            Team names as they appear in `self.label`.
        **kwargs:
            rng : numpy.random.Generator, optional
                Supply your own generator for reproducible draws.

        Returns
        -------
        lambda_h_samples, lambda_a_samples : 1‑D np.ndarray
            Flattened arrays of length  (chains × draws).

        """
        rng: np.random.Generator = (
            np.random.default_rng()
            if kwargs.get("rng") is None
            else kwargs.get("rng")  # type:ignore
        )
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError(f"'rng' must be a numpy.random.Generator or None, got {type(rng)}")

        home_team_id, away_team_id = self.label.transform([home_team, away_team])

        posterior = self.trace.posterior  # xarray Dataset
        intercept = posterior["intercept"].values  # (c,d)
        home_adv = posterior["home"].values[..., home_team_id]
        atts_h = posterior["atts"].values[..., home_team_id]
        atts_a = posterior["atts"].values[..., away_team_id]
        defs_h = posterior["defs"].values[..., home_team_id]
        defs_a = posterior["defs"].values[..., away_team_id]
        beta_hxg = posterior["beta_hxg"].values  # (c,d)
        beta_axg = posterior["beta_axg"].values  # (c,d)
        kappa = posterior["kappa"].values  # (c,d)
        intercept_xg = posterior["intercept_xg"].values  # (c,d)
        theta_h = np.exp(intercept_xg + posterior["home_xg"].values[..., home_team_id])  # (c,d)
        theta_a = np.exp(intercept_xg)  # (c,d)

        n_sample = intercept.shape[0]

        scale_h = kappa / theta_h
        scale_a = kappa / theta_a

        latent_xgh = rng.gamma(shape=kappa, scale=scale_h)  # (c,d)
        latent_xga = rng.gamma(shape=kappa, scale=scale_a)  # (c,d)
        lambda_h = np.exp(
            intercept + home_adv + atts_h + defs_a + beta_hxg * np.log(latent_xgh + 1e-6)
        )
        lambda_a = np.exp(intercept + atts_a + defs_h + beta_axg * np.log(latent_xga + 1e-6))
        lambda_h = lambda_h.ravel()
        lambda_a = lambda_a.ravel()
        prob_H_list = []
        prob_D_list = []
        prob_A_list = []

        for i in range(n_sample):
            home_goals = rng.poisson(lam=lambda_h[i], size=150)
            away_goals = rng.poisson(lam=lambda_a[i], size=150)
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

    def hierarchical_joint_xg_goals(
        self,
        goals_home_obs: np.ndarray,
        goals_away_obs: np.ndarray,
        xg_home_obs: np.ndarray,
        xg_away_obs: np.ndarray,
        home_team: np.ndarray,
        away_team: np.ndarray,
    ):
        with pm.Model():
            # Data
            goals_home = pm.Data("goals_home", goals_home_obs)
            goals_away = pm.Data("goals_away", goals_away_obs)
            xg_home = pm.Data("xg_home", xg_home_obs)
            xg_away = pm.Data("xg_away", xg_away_obs)
            h_idx = pm.Data("home_team", home_team)
            a_idx = pm.Data("away_team", away_team)

            # Shared latent team strengths (sum-to-zero, non-centered)
            tau_att = pm.HalfNormal("tau_att", 1.5)
            tau_def = pm.HalfNormal("tau_def", 1.5)
            raw_att = pm.Normal("raw_att", 0, 1, shape=self.n_teams)
            raw_def = pm.Normal("raw_def", 0, 1, shape=self.n_teams)
            att_u = raw_att * tau_att
            def_u = raw_def * tau_def
            att = pm.Deterministic("att", att_u - pm.math.mean(att_u))
            deff = pm.Deterministic("def", def_u - pm.math.mean(def_u))

            # Shared linear predictor for chance creation
            alpha = pm.Normal("alpha_xg", 0.0, 1.5)  # baseline
            home_adv = pm.Normal("home_adv_xg", 0.0, 0.5)
            eta_h = alpha + home_adv + att[h_idx] - deff[a_idx]
            eta_a = alpha + att[a_idx] - deff[h_idx]

            # xG submodel (Gamma; mean = exp(eta))
            kappa = pm.HalfNormal("kappa_xg", 2.0)  # shape
            theta_xg_h = pm.Deterministic("theta_xg_h", pm.math.exp(eta_h))
            theta_xg_a = pm.Deterministic("theta_xg_a", pm.math.exp(eta_a))
            pm.Gamma("xg_home_like", alpha=kappa, beta=kappa / theta_xg_h, observed=xg_home)
            pm.Gamma("xg_away_like", alpha=kappa, beta=kappa / theta_xg_a, observed=xg_away)

            # Goal submodel: goals ~ Poisson(c * theta_xg^delta)
            alpha_goal = pm.Normal("alpha_goal", 0.0, 1.0)
            delta = pm.TruncatedNormal("delta", 1.0, 0.3, lower=0.0)  # link from xG-rate to goals
            lam_h = pm.Deterministic("lambda_h", pm.math.exp(alpha_goal + delta * eta_h))
            lam_a = pm.Deterministic("lambda_a", pm.math.exp(alpha_goal + delta * eta_a))
            pm.Poisson("home_goals", mu=lam_h, observed=goals_home)
            pm.Poisson("away_goals", mu=lam_a, observed=goals_away)
            trace = pm.sample(
                2000,
                tune=1000,
                cores=os.cpu_count(),
                target_accept=0.95,
                nuts_sampler="numpyro",
                init="adapt_diag_grad",
                return_inferencedata=True,
            )

        self.trace = trace
        return trace
