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
from footix.utils.typing import ProtoBayes


class XGBayesian(ProtoBayes):
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
        xg_home_obs = x_train_cop["fthxg"].to_numpy()
        xg_away_obs = x_train_cop["ftaxg"].to_numpy()
        home_team = x_train_cop["home_team_id"].to_numpy()
        away_team = x_train_cop["away_team_id"].to_numpy()
        self.trace = self.hierarchical_xg_bayes(
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
        posterior = self.trace.posterior
        home = posterior["home"].mean(dim=["chain", "draw"]).values
        intercept = posterior["intercept"].mean(dim=["chain", "draw"]).values
        atts = posterior["atts"].mean(dim=["chain", "draw"]).values
        defs = posterior["defs"].mean(dim=["chain", "draw"]).values
        beta_1 = posterior["beta_hxg"].mean(dim=["chain", "draw"]).values
        beta_2 = posterior["beta_axg"].mean(dim=["chain", "draw"]).values
        # ---------- latent‑xG layer: use their conditional means ----------
        intercept_xg = posterior["intercept_xg"].mean(("chain", "draw")).item()
        home_xg = posterior["home_xg"].mean(("chain", "draw")).values

        theta_h = np.exp(intercept_xg + home_xg[home_team_id])  # mean latent xG for home team
        theta_a = np.exp(intercept_xg)

        home_theta = np.exp(
            intercept
            + home[home_team_id]
            + atts[home_team_id]
            + defs[away_team_id]
            + beta_1 * np.log(theta_h + 1e-6)
        )
        away_theta = np.exp(
            intercept + atts[away_team_id] + defs[home_team_id] + beta_2 * np.log(theta_a + 1e-6)
        )
        return home_theta, away_theta

    def get_samples(
        self, home_team: str, away_team: str, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
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
            np.random.default_rng() if kwargs.get("rng") is None else kwargs.get("rng")
        )  # type:ignore
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError(f"'rng' must be a numpy.random.Generator or None, got {type(rng)}")

        # ------------------------------------------------------------------
        # translate team names → indices
        home_team_id, away_team_id = self.label.transform([home_team, away_team])

        posterior = self.trace.posterior  # xarray Dataset

        # ---------- top‑level parameters (dims: chain, draw) --------------
        intercept = posterior["intercept"].values  # (c,d)
        home_adv = posterior["home"].values[..., home_team_id]
        atts_h = posterior["atts"].values[..., home_team_id]
        atts_a = posterior["atts"].values[..., away_team_id]
        defs_h = posterior["defs"].values[..., home_team_id]
        defs_a = posterior["defs"].values[..., away_team_id]
        beta_hxg = posterior["beta_hxg"].values  # (c,d)
        beta_axg = posterior["beta_axg"].values  # (c,d)

        # ---------- latent‑xG hyper‑parameters ----------------------------
        kappa = posterior["kappa"].values  # (c,d)
        intercept_xg = posterior["intercept_xg"].values  # (c,d)
        theta_h = np.exp(intercept_xg + posterior["home_xg"].values[..., home_team_id])  # (c,d)
        theta_a = np.exp(intercept_xg)  # (c,d)

        # ---------- draw latent xG from Gamma(κ, scale=κ/θ) ----------------
        scale_h = kappa / theta_h
        scale_a = kappa / theta_a

        latent_xgh = rng.gamma(shape=kappa, scale=scale_h)  # (c,d)
        latent_xga = rng.gamma(shape=kappa, scale=scale_a)  # (c,d)

        # ---------- convert to goal‑rate λ --------------------------------
        lambda_h = np.exp(
            intercept + home_adv + atts_h + defs_a + beta_hxg * np.log(latent_xgh + 1e-6)
        )

        lambda_a = np.exp(intercept + atts_a + defs_h + beta_axg * np.log(latent_xga + 1e-6))

        return lambda_h.ravel(), lambda_a.ravel()

    def hierarchical_xg_bayes(
        self,
        goals_home_obs: np.ndarray,
        goals_away_obs: np.ndarray,
        xg_home_obs: np.ndarray,
        xg_away_obs: np.ndarray,
        home_team: np.ndarray,
        away_team: np.ndarray,
    ):
        with pm.Model():
            # Use pm.Data for the observed data and covariates
            goals_home_data = pm.Data("goals_home", goals_home_obs)
            goals_away_data = pm.Data("goals_away", goals_away_obs)
            xg_home_data = pm.Data("xg_home", xg_home_obs)
            xg_away_data = pm.Data("xg_away", xg_away_obs)
            home_team_data = pm.Data("home_team", home_team)
            away_team_data = pm.Data("away_team", away_team)

            # Home advantage and intercept
            home = pm.Normal("home", mu=0, sigma=1, shape=self.n_teams)
            intercept = pm.Normal("intercept", mu=3, sigma=1)

            # Layer A: latent xG predictions
            intercept_xg = pm.Normal("intercept_xg", mu=2, sigma=1)
            home_xg = pm.Normal("home_xg", mu=0, sigma=0.5, shape=self.n_teams)
            theta_h = pm.Deterministic(
                "theta_h", pm.math.exp(intercept_xg + home_xg[home_team_data])
            )
            theta_a = pm.Deterministic("theta_a", pm.math.exp(intercept_xg))
            kappa = pm.HalfNormal("kappa", 2)  # Gamma shape for xG totals
            latent_xgh = pm.Gamma("latent_xgh", kappa, kappa / theta_h, observed=xg_home_data)
            latent_xga = pm.Gamma("latent_xga", kappa, kappa / theta_a, observed=xg_away_data)

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
            beta_hxg = pm.Normal("beta_hxg", 1.0, 0.3)
            beta_axg = pm.Normal("beta_axg", 1.0, 0.3)

            # Calculate theta for home and away
            home_theta = pm.math.exp(
                intercept
                + home[home_team_data]
                + atts[home_team_data]
                + defs[away_team_data]
                + beta_hxg * pm.math.log(latent_xgh[home_team_data] + 1e-6)
            )
            away_theta = pm.math.exp(
                intercept
                + atts[away_team_data]
                + defs[home_team_data]
                + beta_axg * pm.math.log(latent_xga[away_team_data] + 1e-6)
            )

            # Goal likelihood
            pm.Poisson("home_goals", mu=home_theta, observed=goals_home_data)
            pm.Poisson("away_goals", mu=away_theta, observed=goals_away_data)
            # Sample with improved settings
            trace = pm.sample(
                2000,
                tune=1000,
                cores=6,
                target_accept=0.95,
                return_inferencedata=True,
                nuts_sampler="numpyro",
                init="adapt_diag_grad",
            )
        return trace
