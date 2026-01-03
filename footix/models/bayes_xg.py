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
        home_team_id, away_team_id = self.label.transform([home_team, away_team])

        home_goal_expectation, away_goal_expectation = self.goal_expectation(
            home_team_id=home_team_id, away_team_id=away_team_id
        )

        home_probs = stats.poisson.pmf(range(self.n_goals), home_goal_expectation)
        away_probs = stats.poisson.pmf(range(self.n_goals), away_goal_expectation)

        goals_matrix = GoalMatrix(home_probs, away_probs)
        return goals_matrix

    def goal_expectation(self, home_team_id: int, away_team_id: int):
        """Renvoie l'espérance de buts (home, away) pour une affiche (home_team_id, away_team_id)

        à partir du posterior. Si integrate_lognormal=True, intègre le bruit multiplicatif
        log-normal via exp(0.5 * sigma_lambda**2).
        Optionnel: renvoie aussi des quantiles (q) de la distribution postérieure de l'espérance.

        """
        post = self.trace.posterior

        # ======== xG (theta_xg) par affiche ========
        alpha_xg = post["alpha_xg"].values  # (chain, draw)
        home_adv_xg = post["home_adv_xg"].values  # (chain, draw)
        att_xg = post["att_xg"].values  # (chain, draw, n_teams)
        def_xg = post["def_xg"].values  # (chain, draw, n_teams)

        eta_xg_h = alpha_xg + home_adv_xg + att_xg[..., home_team_id] - def_xg[..., away_team_id]
        eta_xg_a = alpha_xg + att_xg[..., away_team_id] - def_xg[..., home_team_id]

        theta_xg_h = np.exp(eta_xg_h)  # (chain, draw)
        theta_xg_a = np.exp(eta_xg_a)

        # ======== composante "goals" ========
        alpha_g = post["alpha_goals"].values  # (chain, draw)
        home_adv_g = post["home_adv_goals"].values  # (chain, draw)
        att_g = post["att_goals"].values  # (chain, draw, n_teams)
        def_g = post["def_goals"].values  # (chain, draw, n_teams)
        gamma_link = post["gamma_link"].values  # (chain, draw)

        base_h = alpha_g + home_adv_g + att_g[..., home_team_id] - def_g[..., away_team_id]
        base_a = alpha_g + att_g[..., away_team_id] - def_g[..., home_team_id]

        log_lam_h = base_h + gamma_link * np.log(theta_xg_h)  # (chain, draw)
        log_lam_a = base_a + gamma_link * np.log(theta_xg_a)

        lam_h_draw = np.exp(log_lam_h)
        lam_a_draw = np.exp(log_lam_a)

        # Moyenne postérieure (chaînes + tirages)
        home_mean = lam_h_draw.mean(axis=(0, 1)).item()
        away_mean = lam_a_draw.mean(axis=(0, 1)).item()

        return home_mean, away_mean

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
        # Team strength parameters
        atts = posterior["atts"].values  # (chains, draws, n_teams)
        defs = posterior["defs"].values  # (chains, draws, n_teams)

        # xG model parameters
        alpha_xg = posterior["alpha_xg"].values  # (chains, draws)
        home_adv_xg = posterior["home_adv_xg"].values  # (chains, draws)

        # Calculate eta (log expected xG)
        eta_h = alpha_xg + home_adv_xg + atts[..., home_team_id] - defs[..., away_team_id]
        eta_a = alpha_xg + atts[..., away_team_id] - defs[..., home_team_id]

        # Calculate goal rates
        lambda_h = np.exp(eta_h)
        lambda_a = np.exp(eta_a)

        # Flatten arrays
        lambda_h = lambda_h.ravel()
        lambda_a = lambda_a.ravel()
        n_samples = len(lambda_h)

        prob_H_list = []
        prob_D_list = []
        prob_A_list = []

        for i in range(n_samples):
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

            chol_att_packed = pm.LKJCholeskyCov(
                "chol_att",
                n=2,
                eta=2.0,
                sd_dist=pm.HalfNormal.dist(5.0),
                compute_corr=False,
            )
            chol_def_packed = pm.LKJCholeskyCov(
                "chol_def",
                n=2,
                eta=2.0,
                sd_dist=pm.HalfNormal.dist(5.0),
                compute_corr=False,
            )
            chol_att = pm.expand_packed_triangular(2, chol_att_packed)
            chol_def = pm.expand_packed_triangular(2, chol_def_packed)
            att_offset = pm.Normal("att_offset", 0.0, 1.0, shape=(self.n_teams, 2))
            att = pm.Deterministic("att", att_offset @ chol_att.T)
            att_xg = pm.Deterministic("att_xg", att[:, 0] - att[:, 0].mean())
            att_goals = pm.Deterministic("att_goals", att[:, 1] - att[:, 1].mean())

            def_offset = pm.Normal("def_offset", 0.0, 1.0, shape=(self.n_teams, 2))

            def_ = pm.Deterministic("def_", def_offset @ chol_def.T)  # (n_teams, 2)
            def_xg = pm.Deterministic("def_xg", def_[:, 0] - def_[:, 0].mean())
            def_goals = pm.Deterministic("def_goals", def_[:, 1] - def_[:, 1].mean())

            # ===== xG MODEL (Gamma with mean = theta_xg, kappa = shape) =====
            alpha_xg = pm.Normal("alpha_xg", 0.0, 1.0)
            home_adv_xg = pm.Normal("home_adv_xg", 0.2, 0.3)
            eta_xg_h = alpha_xg + home_adv_xg + att_xg[h_idx] - def_xg[a_idx]
            eta_xg_a = alpha_xg + 0.0 + att_xg[a_idx] - def_xg[h_idx]

            # log-link for positive rate
            theta_xg_h = pm.Deterministic("theta_xg_h", pm.math.exp(eta_xg_h))
            theta_xg_a = pm.Deterministic("theta_xg_a", pm.math.exp(eta_xg_a))

            # Gamma(mean=theta, shape=kappa) → Gamma(alpha=kappa, beta=kappa/theta)
            kappa = pm.HalfNormal("kappa_xg", 5.0)
            pm.Gamma("xg_home_like", alpha=kappa, beta=kappa / theta_xg_h, observed=xg_home)
            pm.Gamma("xg_away_like", alpha=kappa, beta=kappa / theta_xg_a, observed=xg_away)

            # ===== GOALS MODEL (Poisson with log-link, borrowing strength from xG) =====
            # A stable way to link: log(lambda) = α + H + (att_def terms) + γ * log(theta_xg)
            alpha_goals = pm.Normal("alpha_goals", 0.0, 1.0)
            home_adv_goals = pm.Normal("home_adv_goals", 0.2, 0.3)
            gamma_link = pm.Normal(
                "gamma_link", 1.0, 0.5
            )  # how strongly xG informs goals (elasticity)

            base_h = alpha_goals + home_adv_goals + att_goals[h_idx] - def_goals[a_idx]
            base_a = alpha_goals + 0.0 + att_goals[a_idx] - def_goals[h_idx]

            log_lam_h = base_h + gamma_link * pm.math.log(theta_xg_h)
            log_lam_a = base_a + gamma_link * pm.math.log(theta_xg_a)

            lam_h = pm.Deterministic("lam_h", pm.math.exp(log_lam_h))
            lam_a = pm.Deterministic("lam_a", pm.math.exp(log_lam_a))

            pm.Poisson("home_goals", mu=lam_h, observed=goals_home)
            pm.Poisson("away_goals", mu=lam_a, observed=goals_away)

            trace = pm.sample(
                draws=2000,
                tune=1000,
                cores=min(4, os.cpu_count() or 1),
                target_accept=0.95,
                nuts_sampler="numpyro",
                init="adapt_diag_grad",
            )

        self.trace = trace
        return trace

    def predict_xg(self, home_team: str, away_team: str) -> dict[str, Any]:
        """Return Gamma parameters of home_team and away_team
        Args:
            home_team (str): Name of the home team
            away_team (str): Name of the away team
        """
        res_dict = {"home_team": home_team, "away_team": away_team}
        post = self.trace.posterior
        home_team_id, away_team_id = self.label.transform([home_team, away_team])

        alpha = post["alpha_xg"].mean(("chain", "draw")).item()
        home_adv = post["home_adv_xg"].mean(("chain", "draw")).item()
        att = post["att_xg"].mean(("chain", "draw")).values
        deff = post["def_xg"].mean(("chain", "draw")).values
        kappa_xg = post["kappa_xg"].mean(("chain", "draw")).item()

        eta_h = alpha + home_adv + att[home_team_id] - deff[away_team_id]
        eta_a = alpha + att[away_team_id] - deff[home_team_id]
        theta_h_xg = np.exp(eta_h)
        theta_a_xg = np.exp(eta_a)
        res_dict["alpha"] = kappa_xg
        res_dict["beta_h"] = kappa_xg / theta_h_xg
        res_dict["beta_a"] = kappa_xg / theta_a_xg
        return res_dict
