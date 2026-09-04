from __future__ import annotations

import os
import warnings
from collections.abc import Hashable
from functools import cache
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import scipy.stats as stats

from footix.models.score_matrix import GoalMatrix
from footix.utils.decorators import verify_required_column
from footix.utils.typing import SampleProbaResult

# ---------------------------------------------------------------------------
# Priors of the baseline model (weakly informative, football scale).
#
# All priors are declared on the log-goal scale. The linear predictors are
#
#     log lambda_home = intercept + home + attack_strength[i] - defence_strength[j]
#     log lambda_away = intercept + attack_strength[j] - defence_strength[i]
#
# with the convention: attack_strength > 0 means a better attack and
# defence_strength > 0 means a better defence (fewer goals conceded).
#
# * intercept ~ Normal(log(1.3), 0.3): the prior median rate is
#   exp(log 1.3) = 1.3 goals per team, i.e. ~2.6 goals per match before the
#   home effect. The 95% prior range exp(log(1.3) ± 0.6) is roughly
#   [0.7, 2.4] goals per team.
# * home ~ Normal(0.2, 0.15): global home advantage on the multiplicative
#   scale e^home has prior median e^0.2 ~= 1.22 and a 95% range of about
#   [0.90, 1.65] (a home team scoring between 0.9x and 1.65x its neutral
#   rate). ``home`` is deliberately a single shared parameter: with one
#   season and early walk-forward cutoffs there are too few observations to
#   estimate per-team home effects. A partially pooled variant
#   ``h_i = home + sigma_h * z_i`` can be added later without changing the
#   baseline.
# * sigma_attack, sigma_defence ~ HalfNormal(0.3): typical spread of the
#   sum-to-zero team skills. HalfNormal(0.3) keeps most mass below 0.3*1.96
#   ~= 0.59 on the log scale (a team at +2 sigma_attack scores about
#   exp(0.6) ~= 1.8x its baseline) while remaining weakly informative.
#
# These numbers are a starting point, not sacred; the prior predictive
# checks in tests/test_bayesian_baseline.py guard the plausible scale.
# ---------------------------------------------------------------------------
LOG_BASE_RATE_MU = float(np.log(1.3))
LOG_BASE_RATE_SIGMA = 0.3
HOME_MU = 0.2
HOME_SIGMA = 0.15
SKILL_SIGMA = 0.3

# MCMC diagnostics thresholds -------------------------------------------------
# A fit is considered valid when, on the joint posterior:
#   * max_rhat <= 1.01
#   * no divergent transitions
#   * min bulk and tail ESS >= max(100, 5% of total draws)
# With the default draws=2000 over 4 chains (8000 total draws) that is an
# ESS floor of 400, well above the commonly cited rule of thumb of ~100 per
# chain for stable quantile estimates.
R_HAT_MAX = 1.01
ESS_FRACTION = 0.05
ESS_FLOOR = 100


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


def _log_goal_submodel(
    optional_stats: dict[str, Any],
    home_team: np.ndarray,
    away_team: np.ndarray,
    attack_strength: Any,
    defence_strength: Any,
    n_teams: int,
) -> None:
    """Auxiliary log-shots/SOT/corners channels (EXPERIMENTAL, off by default).

    These channels share the attack/defence skills with the goals
    likelihood and are fed exclusively with match statistics from the
    training window (the same matches that produced the goals), so they
    introduce no temporal leakage. They are kept for future ablation
    studies only; the baseline model is goals-only and the sub-model is not
    meant to be activated in production benchmarks.

    Args:
        optional_stats: Extracted channel dict (see ``_extract_optional_stats_data``).
        home_team: Home team ids, full training vector.
        away_team: Away team ids, full training vector.
        attack_strength: Sum-to-zero attack skill parameter.
        defence_strength: Sum-to-zero defence skill parameter.
    """
    if optional_stats.get("has_shots", False):
        beta_shots = pm.Normal("beta_shots", mu=2.5, sigma=0.5)
        sigma_shots = pm.HalfNormal("sigma_shots", sigma=0.4)
        shots_idx = optional_stats["shots_idx"]
        shots_home_team = home_team[shots_idx]
        shots_away_team = away_team[shots_idx]

        mu_shots_home = (
            beta_shots + attack_strength[shots_home_team] + defence_strength[shots_away_team]
        )
        mu_shots_away = (
            beta_shots + attack_strength[shots_away_team] + defence_strength[shots_home_team]
        )

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
        theta_raw = pm.Normal("theta_raw", mu=0.0, sigma=sigma_theta, shape=n_teams)
        theta = pm.Deterministic("theta", theta_raw - pm.math.mean(theta_raw))
        sot_idx = optional_stats["sot_idx"]
        sot_home_team = home_team[sot_idx]
        sot_away_team = away_team[sot_idx]

        mu_sot_home = (
            beta_sot
            + attack_strength[sot_home_team]
            + defence_strength[sot_away_team]
            + theta[sot_home_team]
        )
        mu_sot_away = (
            beta_sot
            + attack_strength[sot_away_team]
            + defence_strength[sot_home_team]
            + theta[sot_away_team]
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

        mu_corners_home = (
            beta_corners + attack_strength[corners_home_team] + defence_strength[corners_away_team]
        )
        mu_corners_away = (
            beta_corners + attack_strength[corners_away_team] + defence_strength[corners_home_team]
        )

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


def _generative_model(
    goals_home_obs: np.ndarray,
    goals_away_obs: np.ndarray,
    home_team: np.ndarray,
    away_team: np.ndarray,
    n_teams: int,
    use_stats: bool,
    optional_stats: dict[str, Any] | None,
) -> pm.Model:
    """Build the baseline generative model (priors + likelihoods).

    Baseline (goals only):

    .. math::

        a_i \\sim N(0, \\sigma_a), \\quad d_i \\sim N(0, \\sigma_d)
        \\text{ (sum-to-zero, non-centered)}
        \\log \\lambda_H = \\mu + h + a_i - d_j
        \\log \\lambda_A = \\mu + a_j - d_i
        Y_H \\sim Poisson(\\lambda_H), \\quad Y_A \\sim Poisson(\\lambda_A)

    with ``mu ~ N(log 1.3, 0.3)``, global ``h ~ N(0.2, 0.15)`` and
    ``sigma_a, sigma_d ~ HalfNormal(0.3)``. ``attack_strength > 0`` means a
    better attack, ``defence_strength > 0`` means a better defence (the
    minus sign above).

    The observed goals are registered as ``pm.Data`` nodes so the same
    model definition is reused for prior predictive sampling.

    Args:
        goals_home_obs: Observed home goals (length n_matches).
        goals_away_obs: Observed away goals.
        home_team: Home team ids.
        away_team: Away team ids.
        n_teams: Number of distinct teams.
        use_stats: Whether to add the experimental auxiliary channels.
        optional_stats: Extracted channels for ``use_stats``.

    Returns:
        The PyMC model.
    """
    optional_stats = optional_stats or {}
    with pm.Model() as model:
        goals_home_data = pm.Data("goals_home", goals_home_obs)
        goals_away_data = pm.Data("goals_away", goals_away_obs)
        home_team_data = pm.Data("home_team", home_team)
        away_team_data = pm.Data("away_team", away_team)

        # Baseline goal rate and GLOBAL home advantage (see module docstring).
        intercept = pm.Normal("intercept", mu=LOG_BASE_RATE_MU, sigma=LOG_BASE_RATE_SIGMA)
        home = pm.Normal("home", mu=HOME_MU, sigma=HOME_SIGMA)

        # Attack ratings: non-centered, sum-to-zero, HalfNormal dispersion.
        sigma_attack = pm.HalfNormal("sigma_attack", sigma=SKILL_SIGMA)
        raw_attack = pm.Normal("raw_attack", mu=0, sigma=1, shape=n_teams)
        attack_strength = pm.Deterministic(
            "attack_strength",
            (raw_attack - pm.math.mean(raw_attack)) * sigma_attack,
        )
        # Defence ratings: same parameterization; LARGER = BETTER defence,
        # hence the minus sign in the linear predictors below.
        sigma_defence = pm.HalfNormal("sigma_defence", sigma=SKILL_SIGMA)
        raw_defence = pm.Normal("raw_defence", mu=0, sigma=1, shape=n_teams)
        defence_strength = pm.Deterministic(
            "defence_strength",
            (raw_defence - pm.math.mean(raw_defence)) * sigma_defence,
        )

        home_theta = pm.math.exp(
            intercept + home + attack_strength[home_team_data] - defence_strength[away_team_data]
        )
        away_theta = pm.math.exp(
            intercept + attack_strength[away_team_data] - defence_strength[home_team_data]
        )
        pm.Poisson("home_goals", mu=home_theta, observed=goals_home_data)
        pm.Poisson("away_goals", mu=away_theta, observed=goals_away_data)

        if use_stats:
            _log_goal_submodel(
                optional_stats,
                home_team_data.eval(),
                away_team_data.eval(),
                attack_strength,
                defence_strength,
                n_teams,
            )
    return model


def prior_predictive_draws(
    n_teams: int,
    n_matches: int,
    draws: int = 2000,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample goal counts from the generative prior (no data conditioning).

    Draws from the same model definition used by ``BayesianModel.fit``, so
    the prior predictive checks cannot drift away from the fitted priors.

    Args:
        n_teams: Number of teams in the synthetic league.
        n_matches: Number of synthetic matches.
        draws: Number of prior predictive draws.
        random_seed: Seed for reproducibility.

    Returns:
        Tuple ``(home_goals, away_goals)`` of shape ``(draws, n_matches)``.
    """
    home_team = np.arange(n_matches) % n_teams
    away_team = (np.arange(n_matches) + 1) % n_teams
    model = _generative_model(
        goals_home_obs=np.zeros(n_matches, dtype=int),
        goals_away_obs=np.zeros(n_matches, dtype=int),
        home_team=home_team,
        away_team=away_team,
        n_teams=n_teams,
        use_stats=False,
        optional_stats=None,
    )
    with model:
        prior = pm.sample_prior_predictive(draws=draws, random_seed=random_seed)
    home = prior.prior_predictive["home_goals"].stack(sample=("chain", "draw")).values
    away = prior.prior_predictive["away_goals"].stack(sample=("chain", "draw")).values
    return home.astype(float), away.astype(float)


class BayesianModel:
    """Bayesian hierarchical model for football scores using Poisson likelihoods.

    The baseline model is goals-only: a global intercept (base goal rate),
    a global home advantage and sum-to-zero attack/defence skills shared by
    the home and away Poisson likelihoods. ``use_stats=True`` additionally
    fits the EXPERIMENTAL shots/SOT/corners channels (kept for future
    ablations, never enabled in the baseline).

    Point predictions are posterior predictive means: every market
    probability returned by ``predict`` is the average over posterior draws
    of the per-draw market probability, i.e. :math:`\\hat p_k =
    \\frac{1}{S} \\sum_s p_k^{(s)}`.

    Attributes
    ----------
    n_teams : int | None
        Number of distinct teams in the league (inferred on first fit).
    n_goals : int
        Maximum number of goals considered when computing score probabilities.
    trace : arviz.InferenceData | None
        Posterior samples after calling `fit`. None until the model is fitted.

    """

    def __init__(
        self,
        n_goals: int,
        n_teams: int | None = None,
        use_stats: bool = False,
        random_seed: int | None = 42,
    ):
        self.n_teams = n_teams
        self.n_goals = n_goals
        self.use_stats = use_stats
        self.random_seed = random_seed
        self.trace: az.InferenceData | None = None
        self._team_to_id: dict[Hashable, int] = {}
        self._diagnostics: dict[str, Any] | None = None

    @verify_required_column(column_names={"home_team", "away_team", "fthg", "ftag"})
    def fit(self, X_train: pd.DataFrame, sample_kwargs: dict[str, Any] | None = None):
        """Fit the model on training data (strictly past matches only).

        Any cached prediction from a previous fit is dropped: every fit
        invalidates ``predict``/``get_samples``/``get_market_samples`` caches
        so a refitted instance can never serve stale posterior values.

        Args:
            X_train: Training dataframe with columns home_team, away_team,
                fthg, ftag (football-data conventions).
            sample_kwargs: Optional overrides for ``pm.sample`` kwargs.
        """
        self.get_samples.cache_clear()
        self.get_market_samples.cache_clear()
        self._diagnostics = None
        x_train_cop = X_train.copy(deep=False)
        teams = pd.concat([X_train["home_team"], X_train["away_team"]]).unique()
        if self.n_teams is None:
            self.n_teams = len(teams)
        elif self.n_teams != len(teams):
            raise ValueError(
                f"Teams in training data do not match the initialized teams. "
                f"Expected: {self.n_teams}, got: {teams}."
            )

        self._team_to_id = {team: team_id for team_id, team in enumerate(sorted(teams))}
        x_train_cop["home_team_id"] = X_train["home_team"].map(self._team_to_id)
        x_train_cop["away_team_id"] = X_train["away_team"].map(self._team_to_id)

        # Series.map can return float64 even with all-int values; PyMC
        # requires integer dtypes for indexing (advanced_subtensor).
        goals_home_obs = x_train_cop["fthg"].to_numpy(dtype=float)
        goals_away_obs = x_train_cop["ftag"].to_numpy(dtype=float)
        home_team = x_train_cop["home_team_id"].to_numpy(dtype=np.int64)
        away_team = x_train_cop["away_team_id"].to_numpy(dtype=np.int64)
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

    def predict(self, home_team: Hashable, away_team: Hashable) -> GoalMatrix:
        """Posterior-predictive score matrix for one match.

        The marginals are the average over posterior draws of the per-draw
        truncated Poisson marginals. Because 1X2 and O/U probabilities are
        linear masks of the joint score matrix, ``predict`` is numerically
        equal to ``get_market_samples(...).mean(axis=0)`` — there is only
        one definition of the model probabilities.

        Args:
            home_team: Home team name.
            away_team: Away team name.

        Returns:
            GoalMatrix whose 1X2/O-U probabilities are posterior predictive
            means.
        """
        joint = self.posterior_predictive_distribution(home_team, away_team)
        home_mean = joint.sum(axis=1)
        away_mean = joint.sum(axis=0)
        outer = np.outer(home_mean, away_mean)
        # High-goal cells can underflow to exactly 0 for very small rates;
        # joint is 0 there too, so a ratio of 1 keeps the cell at 0 instead
        # of producing a NaN from 0/0.
        correlation = np.divide(joint, outer, out=np.ones_like(joint), where=outer > 0)
        return GoalMatrix(home_mean, away_mean, correlation_matrix=correlation)

    def goal_expectation(self, home_team_id: int, away_team_id: int) -> tuple[float, float]:
        """Posterior predictive expected goal rates for one fixture.

        Returns :math:`E[\\lambda \\mid D]`, the average over posterior draws
        of the per-draw rates :math:`\\lambda^{(s)} = \\exp(\\eta^{(s)})`.
        This is deliberately NOT ``exp`` of the posterior mean linear
        predictor: ``exp`` is convex, so ``exp(E[eta])`` is biased low
        relative to ``E[exp(eta)]`` (Jensen). Probabilities come from
        ``predict``/``get_market_samples``; the rates returned here are the
        matching expected goals of that posterior predictive distribution.

        Args:
            home_team_id: Home team id.
            away_team_id: Away team id.

        Returns:
            Tuple ``(expected_home_goals, expected_away_goals)``.
        """
        mu_home, mu_away = self._draw_goal_rates([home_team_id], [away_team_id])
        return float(mu_home[:, 0].mean()), float(mu_away[:, 0].mean())

    def posterior_predictive_distribution(
        self, home_team: Hashable, away_team: Hashable
    ) -> np.ndarray:
        """Posterior predictive joint score distribution (n_goals x n_goals).

        The mean over posterior draws of the per-draw truncated joint score
        matrix. Every market probability is a linear mask of this matrix, so
        averaging it once makes ``predict`` and ``get_market_samples`` agree
        exactly. Unlike an outer product of the averaged marginals, this
        preserves the per-draw covariance, so score-level checks
        (P(0-0), P(1-1), ...) must use this matrix.
        """
        home_id = self._team_to_id[home_team]
        away_id = self._team_to_id[away_team]
        home_pmf, away_pmf = self._draw_marginals([home_id], [away_id])
        return (home_pmf[:, 0, :, None] * away_pmf[:, 0, None, :]).mean(axis=0)

    @cache
    def get_samples(
        self, home_team: Hashable, away_team: Hashable, **kwargs: Any
    ) -> SampleProbaResult:
        """Posterior probability draws for the 1X2 market.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            **kwargs: Ignored; accepted for API compatibility.

        Returns:
            ``SampleProbaResult`` with per-draw home/draw/away probabilities.
        """
        if kwargs:
            warnings.warn(
                f"Ignoring unexpected keyword arguments: {list(kwargs.keys())}", stacklevel=2
            )

        probabilities = self.get_market_samples(home_team, away_team, "1X2")
        return SampleProbaResult(
            proba_home=probabilities[:, 0],
            proba_draw=probabilities[:, 1],
            proba_away=probabilities[:, 2],
        )

    @cache
    def get_market_samples(
        self, home_team: Hashable, away_team: Hashable, market: str
    ) -> np.ndarray:
        """Return posterior probability samples for 1X2 or a total-goals market.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            market: "1X2" or "O/U2.5" (aliases "U2.5"/"O2.5").

        Returns:
            Array of shape ``(n_draws, n_outcomes)``.
        """
        if market not in {"1X2", "O/U2.5", "U2.5", "O2.5"}:
            raise ValueError(f"Unsupported market: {market}")

        home_team_id = self._team_to_id[home_team]
        away_team_id = self._team_to_id[away_team]
        return self._market_probs([home_team_id], [away_team_id], market)[:, 0, :]

    def get_market_samples_batch(
        self, matches: list[tuple[Hashable, Hashable]], market: str
    ) -> np.ndarray:
        """Return aligned posterior probability samples for several matches.

        Every row of the returned array is computed from the *same* posterior
        draw, so probabilities across matches can be combined jointly, e.g. to
        simulate correlated portfolio scenarios. Per-match calls to
        ``get_market_samples`` iterate the posterior in the same order, but
        this batch method makes the shared draw axis explicit.

        Args:
            matches: List of ``(home_team, away_team)`` pairs.
            market: "1X2" or "O/U2.5".

        Returns:
            Array of shape ``(n_draws, n_matches, n_outcomes)`` where axis 0
            is the shared posterior draw index and ``n_outcomes`` is 3 for
            "1X2" and 2 (under, over) otherwise.
        """
        if market not in {"1X2", "O/U2.5", "U2.5", "O2.5"}:
            raise ValueError(f"Unsupported market: {market}")

        home_ids = [self._team_to_id[home] for home, _ in matches]
        away_ids = [self._team_to_id[away] for _, away in matches]
        return self._market_probs(home_ids, away_ids, market)

    def _draw_goal_rates(
        self, home_ids: list[int], away_ids: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-draw goal rates ``(n_draws, n_matches)`` for the given fixtures.

        ``mu_home[s] = exp(intercept[s] + home[s] + attack[i,s] - defence[j,s])``
        and symmetrically for the away side, straight from the stacked
        posterior draws.
        """
        if self.trace is None:
            raise RuntimeError("fit() must be called before predicting")
        posterior = self.trace.posterior
        attack = posterior["attack_strength"].stack(sample=("chain", "draw")).values
        defence = posterior["defence_strength"].stack(sample=("chain", "draw")).values
        intercept = posterior["intercept"].stack(sample=("chain", "draw")).values
        home = posterior["home"].stack(sample=("chain", "draw")).values
        home_ids = np.asarray(home_ids, dtype=int)
        away_ids = np.asarray(away_ids, dtype=int)

        mu_home = np.exp(
            intercept[:, None] + home[:, None] + attack[home_ids].T - defence[away_ids].T
        )
        mu_away = np.exp(intercept[:, None] + attack[away_ids].T - defence[home_ids].T)
        return mu_home, mu_away

    def _draw_marginals(
        self, home_ids: list[int], away_ids: list[int], chunk_size: int = 512
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-draw truncated, renormalized Poisson marginals.

        Every posterior draw yields its own ``(mu_home, mu_away)`` pair;
        each pair is truncated at ``n_goals`` and renormalized exactly like
        ``GoalMatrix`` does, so ``predict`` and the sample methods share one
        consistent definition and no Monte Carlo noise is involved.

        Args:
            home_ids: Home team ids, one per match.
            away_ids: Away team ids, one per match.
            chunk_size: Draws processed per vectorized chunk.

        Returns:
            Tuple ``(home_pmf, away_pmf)`` of shape
            ``(n_draws, n_matches, n_goals)``.
        """
        mu_home, mu_away = self._draw_goal_rates(home_ids, away_ids)
        n_draws, n_matches = mu_home.shape
        ks = np.arange(self.n_goals)
        home_pmf = np.empty((n_draws, n_matches, self.n_goals), dtype=float)
        away_pmf = np.empty((n_draws, n_matches, self.n_goals), dtype=float)

        for start in range(0, n_draws, chunk_size):
            stop = min(start + chunk_size, n_draws)
            home_chunk = stats.poisson.pmf(ks[None, None, :], mu_home[start:stop, :, None])
            away_chunk = stats.poisson.pmf(ks[None, None, :], mu_away[start:stop, :, None])
            home_chunk /= home_chunk.sum(axis=-1, keepdims=True)
            away_chunk /= away_chunk.sum(axis=-1, keepdims=True)
            home_pmf[start:stop] = home_chunk
            away_pmf[start:stop] = away_chunk

        return home_pmf, away_pmf

    def _market_probs(
        self,
        home_ids: list[int],
        away_ids: list[int],
        market: str,
        chunk_size: int = 512,
    ) -> np.ndarray:
        """Exact per-draw market probabilities from the truncated score matrix.

        Args:
            home_ids: Home team ids, one per match.
            away_ids: Away team ids, one per match.
            market: "1X2" or a total-goals market.
            chunk_size: Draws processed per vectorized chunk.

        Returns:
            Array of shape ``(n_draws, n_matches, n_outcomes)``.
        """
        home_pmf, away_pmf = self._draw_marginals(home_ids, away_ids, chunk_size)
        n_draws, n_matches, _ = home_pmf.shape
        ks = np.arange(self.n_goals)

        if market == "1X2":
            masks: tuple[np.ndarray, ...] = (
                np.tril(np.ones((self.n_goals, self.n_goals)), -1),
                np.eye(self.n_goals),
                np.triu(np.ones((self.n_goals, self.n_goals)), 1),
            )
            n_outcomes = 3
        else:
            over = (ks[:, None] + ks[None, :]) >= 3
            masks = (np.logical_not(over), over)
            n_outcomes = 2

        result = np.empty((n_draws, n_matches, n_outcomes), dtype=float)
        for start in range(0, n_draws, chunk_size):
            stop = min(start + chunk_size, n_draws)
            matrix = home_pmf[start:stop, :, :, None] * away_pmf[start:stop, :, None, :]
            for outcome, mask in enumerate(masks):
                probs = np.sum(matrix * mask[None, None], axis=(2, 3))
                result[start:stop, :, outcome] = probs

        return result

    def hierarchical_bayes(
        self,
        goals_home_obs: np.ndarray,
        goals_away_obs: np.ndarray,
        home_team: np.ndarray,
        away_team: np.ndarray,
        optional_stats: dict[str, Any] | None = None,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> az.InferenceData:
        """Fit the ``_generative_model`` baseline with NUTS (nutpie).

        Args:
            goals_home_obs: Observed home goals.
            goals_away_obs: Observed away goals.
            home_team: Home team ids.
            away_team: Away team ids.
            optional_stats: Extracted auxiliary channels (use_stats only).
            sample_kwargs: Optional overrides for ``pm.sample`` kwargs.

        Returns:
            The posterior InferenceData.
        """
        model = _generative_model(
            goals_home_obs,
            goals_away_obs,
            home_team,
            away_team,
            n_teams=self.n_teams,  # type:ignore[arg-type]
            use_stats=self.use_stats,
            optional_stats=optional_stats,
        )
        with model:
            inference_kwargs: dict[str, Any] = {
                "draws": 2000,
                "tune": 1000,
                "cores": min(4, os.cpu_count() or 1),
                "nuts_sampler": "nutpie",
                "target_accept": 0.95,
                "return_inferencedata": True,
            }
            if sample_kwargs is not None:
                inference_kwargs.update(sample_kwargs)
            if self.random_seed is not None:
                inference_kwargs.setdefault("random_seed", self.random_seed)

            trace = pm.sample(**inference_kwargs)
        return trace

    def get_diagnostics(self) -> dict[str, Any]:
        """MCMC convergence diagnostics for the last fit.

        Computes (once per fit, cached until the next ``fit``):

        * ``divergences``: number of divergent transitions;
        * ``max_rhat``: worst (largest) split-R-hat over all posterior vars;
        * ``min_ess_bulk`` / ``min_ess_tail``: worst effective sample sizes;
        * ``status``: ``"valid"``, ``"invalid_mcmc"`` or ``"failed"``.

        A fit is ``valid`` when ``max_rhat <= 1.01``, ``divergences == 0``
        and both ESS values are at least ``max(100, 5% of total draws)``
        (400 with the default 4 x 2000 draws). Predictions of an
        ``invalid_mcmc`` fit must never be compared as if they came from a
        converged posterior; the walk-forward evaluator drops such windows.

        Returns:
            A JSON-serializable dictionary.
        """
        if self._diagnostics is not None:
            return dict(self._diagnostics)
        if self.trace is None:
            self._diagnostics = {"status": "failed", "reason": "model not fitted"}
            return dict(self._diagnostics)

        try:
            summary = az.summary(self.trace.posterior, kind="diagnostics")
            divergences = int(self.trace.sample_stats["diverging"].sum().item())
            max_rhat = float(np.max(summary["r_hat"].to_numpy()))
            min_ess_bulk = float(np.min(summary["ess_bulk"].to_numpy()))
            min_ess_tail = float(np.min(summary["ess_tail"].to_numpy()))
        except Exception as exc:  # pragma: no cover - defensive
            self._diagnostics = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            return dict(self._diagnostics)

        n_total_draws = self.trace.posterior.sizes["chain"] * self.trace.posterior.sizes["draw"]
        ess_threshold = max(ESS_FLOOR, int(ESS_FRACTION * n_total_draws))

        violations: list[str] = []
        if not np.isfinite(max_rhat):
            violations.append("non-finite max_rhat")
        elif max_rhat > R_HAT_MAX:
            violations.append(f"max_rhat={max_rhat:.3f} > {R_HAT_MAX}")
        if divergences > 0:
            violations.append(f"divergences={divergences}")
        if not np.isfinite(min_ess_bulk) or min_ess_bulk < ess_threshold:
            violations.append(f"min_ess_bulk={min_ess_bulk:.0f} < {ess_threshold}")
        if not np.isfinite(min_ess_tail) or min_ess_tail < ess_threshold:
            violations.append(f"min_ess_tail={min_ess_tail:.0f} < {ess_threshold}")

        self._diagnostics = {
            "status": "valid" if not violations else "invalid_mcmc",
            "divergences": divergences,
            "max_rhat": max_rhat,
            "min_ess_bulk": min_ess_bulk,
            "min_ess_tail": min_ess_tail,
            "ess_threshold": ess_threshold,
            "reason": "; ".join(violations) if violations else None,
        }
        return dict(self._diagnostics)
