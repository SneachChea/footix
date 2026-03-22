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


def _resolve_column_name(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Resolve the first available column name among candidates.

    Args:
        df: Input dataframe.
        candidates: Candidate column names in priority order.

    Returns:
        The first matching column name or None.
    """
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _extract_optional_stats_data(df: pd.DataFrame) -> dict[str, Any]:
    """Extract optional match-statistics channels with per-channel validity masks.

    Args:
        df: Training dataframe.

    Returns:
        Dict with channel flags and masked arrays for shots, shots on target, and corners.
    """
    result: dict[str, Any] = {
        "has_shots": False,
        "has_sot": False,
        "has_corners": False,
        "shots_idx": None,
        "shots_home_log": None,
        "shots_away_log": None,
        "sot_idx": None,
        "sot_home_log": None,
        "sot_away_log": None,
        "corners_idx": None,
        "corners_home_log": None,
        "corners_away_log": None,
    }

    channel_specs = [
        (
            "shots",
            ("hs", "HS"),
            ("as", "AS"),
            "shots_idx",
            "shots_home_log",
            "shots_away_log",
            "has_shots",
        ),
        (
            "sot",
            ("hst", "HST"),
            ("ast", "AST"),
            "sot_idx",
            "sot_home_log",
            "sot_away_log",
            "has_sot",
        ),
        (
            "corners",
            ("hc", "HC"),
            ("ac", "AC"),
            "corners_idx",
            "corners_home_log",
            "corners_away_log",
            "has_corners",
        ),
    ]

    for (
        _,
        home_candidates,
        away_candidates,
        idx_key,
        home_key,
        away_key,
        flag_key,
    ) in channel_specs:
        home_col = _resolve_column_name(df, home_candidates)
        away_col = _resolve_column_name(df, away_candidates)

        if home_col is None or away_col is None:
            continue

        home_vals = pd.to_numeric(df[home_col], errors="coerce").to_numpy(dtype=np.float64)
        away_vals = pd.to_numeric(df[away_col], errors="coerce").to_numpy(dtype=np.float64)
        valid_mask = np.isfinite(home_vals) & np.isfinite(away_vals)

        if not np.any(valid_mask):
            continue

        valid_idx = np.where(valid_mask)[0].astype(np.int64)
        home_clipped = np.clip(home_vals[valid_mask], a_min=0.5, a_max=None)
        away_clipped = np.clip(away_vals[valid_mask], a_min=0.5, a_max=None)

        result[idx_key] = valid_idx
        result[home_key] = np.log(home_clipped + 1e-5)
        result[away_key] = np.log(away_clipped + 1e-5)
        result[flag_key] = True

    return result


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

    def __init__(
        self,
        n_goals: int,
        n_teams: int | None = None,
        calibrate: bool = False,
        use_stats: bool = False,
    ):
        self.n_teams = n_teams
        self.n_goals = n_goals
        self.calibrate = calibrate
        self.use_stats = use_stats
        self.trace: az.InferenceData | None = None
        self.label = preprocessing.LabelEncoder()

    @verify_required_column(column_names={"home_team", "away_team", "fthg", "ftag"})
    def fit(self, X_train: pd.DataFrame, sample_kwargs: dict[str, Any] | None = None):
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
        optional_stats = _extract_optional_stats_data(x_train_cop) if self.use_stats else None
        hierarchical_kwargs: dict[str, Any] = {"optional_stats": optional_stats}
        if sample_kwargs is not None:
            hierarchical_kwargs["sample_kwargs"] = sample_kwargs

        self.trace = self.hierarchical_bayes(
            goals_home_obs,
            goals_away_obs,
            home_team,
            away_team,
            **hierarchical_kwargs,
        )

    @cache
    def _posterior_means(self) -> dict[str, np.ndarray]:
        p = self.trace.posterior  # type:ignore
        self._cached_means = {
            "home": p["home"].mean(("chain", "draw")).values,
            "intercept": p["intercept"].mean(("chain", "draw")).values.item(),
            "atts": p["atts"].mean(("chain", "draw")).values,
            "defs": p["defs"].mean(("chain", "draw")).values,
        }
        if self.calibrate:
            self._cached_means["tau"] = p["tau"].mean(("chain", "draw")).values.item()
            self._cached_means["bias"] = p["bias"].mean(("chain", "draw")).values
        return self._cached_means

    def _apply_calibration(
        self, probs: np.ndarray, tau: np.ndarray | float, bias: np.ndarray
    ) -> np.ndarray:
        """Apply calibration transformation to match outcome probabilities.

        Parameters
        ----------
        probs : np.ndarray
            Raw probabilities for [Home, Draw, Away], shape (3,) or (n, 3)
        tau : float
            Temperature parameter for scaling
        bias : np.ndarray
            Class-wise bias for [Home, Draw, Away], shape (3,)

        Returns
        -------
        np.ndarray
            Calibrated probabilities with same shape as input
        """
        eps = 1e-12
        probs = np.clip(probs, eps, 1.0)

        # Convert to logits
        logits = np.log(probs + eps)

        # Apply calibration: tau * logits + bias
        calibrated_logits = tau * logits + bias

        # Softmax to get calibrated probabilities
        exp_logits = np.exp(calibrated_logits - np.max(calibrated_logits, axis=-1, keepdims=True))
        calibrated_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return calibrated_probs

    def predict(self, home_team: str, away_team: str) -> GoalMatrix:
        home_id, away_id = self.label.transform([home_team, away_team])

        home_mu, away_mu = self.goal_expectation(home_team_id=home_id, away_team_id=away_id)

        ks = np.arange(self.n_goals)
        home_pmf = stats.poisson.pmf(ks, home_mu)
        away_pmf = stats.poisson.pmf(ks, away_mu)

        goal_matrix = GoalMatrix(home_pmf, away_pmf)

        # Apply calibration if enabled
        if self.calibrate:
            # Get raw match probabilities
            raw_probas = goal_matrix.return_probas()
            raw_probs = np.array(
                [raw_probas.proba_home, raw_probas.proba_draw, raw_probas.proba_away]
            )

            # Get calibration parameters and apply transformation
            means = self._posterior_means()
            calibrated_probs = self._apply_calibration(raw_probs, means["tau"], means["bias"])

            # Create a correlation matrix that adjusts the outcome probabilities
            # to match the calibrated values while preserving the goal distribution shape
            home_win_mask = np.tril(np.ones((self.n_goals, self.n_goals)), -1)
            draw_mask = np.diag(np.ones(self.n_goals))
            away_win_mask = np.triu(np.ones((self.n_goals, self.n_goals)), 1)

            # Calculate scaling factors for each outcome
            scale_home = calibrated_probs[0] / (raw_probs[0] + 1e-12)
            scale_draw = calibrated_probs[1] / (raw_probs[1] + 1e-12)
            scale_away = calibrated_probs[2] / (raw_probs[2] + 1e-12)

            # Build correlation matrix
            correlation_matrix = (
                scale_home * home_win_mask + scale_draw * draw_mask + scale_away * away_win_mask
            )

            # Create new GoalMatrix with correlation applied
            goal_matrix = GoalMatrix(home_pmf, away_pmf, correlation_matrix=correlation_matrix)

        return goal_matrix

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

        # Extract calibration parameters if enabled
        if self.calibrate:
            tau_samples = _posterior["tau"].stack(sample=("chain", "draw")).values
            bias_samples = _posterior["bias"].stack(sample=("chain", "draw")).values

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

            # Apply calibration if enabled
            if self.calibrate:
                raw_probs = np.array([prob_H, prob_D, prob_A])
                calibrated_probs = self._apply_calibration(
                    raw_probs, tau_samples[i], bias_samples[:, i]
                )
                prob_H, prob_D, prob_A = (
                    calibrated_probs[0],
                    calibrated_probs[1],
                    calibrated_probs[2],
                )

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
        optional_stats: dict[str, Any] | None = None,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> az.InferenceData:
        optional_stats = optional_stats or {}
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
            intercept = pm.Normal("intercept", mu=3, sigma=1)

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

            # Optional auxiliary channels from match statistics
            if optional_stats.get("has_shots", False):
                beta_shots = pm.Normal("beta_shots", mu=2.5, sigma=0.5)
                sigma_shots = pm.HalfNormal("sigma_shots", sigma=0.4)
                shots_idx = optional_stats["shots_idx"]
                shots_home_team = home_team[shots_idx]
                shots_away_team = away_team[shots_idx]

                mu_shots_home = beta_shots + atts[shots_home_team] + defs[shots_away_team]
                mu_shots_away = beta_shots + atts[shots_away_team] + defs[shots_home_team]

                pm.Normal(
                    "log_shots_home_obs",
                    mu=mu_shots_home,
                    sigma=sigma_shots,
                    observed=optional_stats["shots_home_log"],
                )
                pm.Normal(
                    "log_shots_away_obs",
                    mu=mu_shots_away,
                    sigma=sigma_shots,
                    observed=optional_stats["shots_away_log"],
                )

            if optional_stats.get("has_sot", False):
                beta_sot = pm.Normal("beta_sot", mu=1.5, sigma=0.5)
                sigma_sot = pm.HalfNormal("sigma_sot", sigma=0.4)
                sigma_theta = pm.HalfNormal("sigma_theta", sigma=0.2)
                theta_raw = pm.Normal("theta_raw", mu=0.0, sigma=sigma_theta, shape=self.n_teams)
                theta = pm.Deterministic("theta", theta_raw - pm.math.mean(theta_raw))
                sot_idx = optional_stats["sot_idx"]
                sot_home_team = home_team[sot_idx]
                sot_away_team = away_team[sot_idx]

                mu_sot_home = (
                    beta_sot + atts[sot_home_team] + defs[sot_away_team] + theta[sot_home_team]
                )
                mu_sot_away = (
                    beta_sot + atts[sot_away_team] + defs[sot_home_team] + theta[sot_away_team]
                )

                pm.Normal(
                    "log_sot_home_obs",
                    mu=mu_sot_home,
                    sigma=sigma_sot,
                    observed=optional_stats["sot_home_log"],
                )
                pm.Normal(
                    "log_sot_away_obs",
                    mu=mu_sot_away,
                    sigma=sigma_sot,
                    observed=optional_stats["sot_away_log"],
                )

            if optional_stats.get("has_corners", False):
                beta_corners = pm.Normal("beta_corners", mu=1.5, sigma=0.5)
                sigma_corners = pm.HalfNormal("sigma_corners", sigma=0.5)
                corners_idx = optional_stats["corners_idx"]
                corners_home_team = home_team[corners_idx]
                corners_away_team = away_team[corners_idx]

                mu_corners_home = beta_corners + atts[corners_home_team] + defs[corners_away_team]
                mu_corners_away = beta_corners + atts[corners_away_team] + defs[corners_home_team]

                pm.Normal(
                    "log_corners_home_obs",
                    mu=mu_corners_home,
                    sigma=sigma_corners,
                    observed=optional_stats["corners_home_log"],
                )
                pm.Normal(
                    "log_corners_away_obs",
                    mu=mu_corners_away,
                    sigma=sigma_corners,
                    observed=optional_stats["corners_away_log"],
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

                # Calibration layer: temperature scaling + class-wise bias
                tau = pm.Normal("tau", mu=1.0, sigma=0.3)  # temperature parameter
                bias = pm.Normal("bias", mu=0.0, sigma=0.3, shape=3)  # class-wise bias (H, D, A)

                # Apply calibration: tau * log(probs) + bias, then softmax
                logit_raw = pm.math.log(match_probs_raw + eps)  # safe log
                calibrated_logits = tau * logit_raw + bias
                match_probs = pm.Deterministic(
                    "match_probs", pt.special.softmax(calibrated_logits, axis=1)
                )

                pm.Categorical("match_outcomes", p=match_probs, observed=match_res_data)
                inference_kwargs: dict[str, Any] = {
                    "draws": 2000,
                    "tune": 1000,
                    "cores": min(4, os.cpu_count() or 1),
                    "init": "adapt_diag_grad",
                    "target_accept": 0.95,
                    "return_inferencedata": True,
                }
                if sample_kwargs is not None:
                    inference_kwargs.update(sample_kwargs)
                trace = pm.sample(**inference_kwargs)
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
