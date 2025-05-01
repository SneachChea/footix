from copy import copy

import numpy as np
import pandas as pd
import pymc as pm
import scipy.stats as stats
from sklearn import preprocessing
from footix.models.utils import process_game_week_index
from footix.utils.typing import ProtoBayes
from footix.models.score_matrix import GoalMatrix
from footix.utils.decorators import verify_required_column


class Bayesian(ProtoBayes):
    def __init__(self, n_teams: int, n_goals: int):
        self.n_teams = n_teams
        self.n_goals = n_goals
        self.label = preprocessing.LabelEncoder()

    @verify_required_column(column_names={"HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"})
    def fit(self, X_train: pd.DataFrame):
        x_train_cop = copy(X_train)
        self.label.fit(X_train["HomeTeam"])  # type: ignore
        x_train_cop["HomeTeamId"] = self.label.transform(X_train["HomeTeam"])
        x_train_cop["AwayTeamId"] = self.label.transform(X_train["AwayTeam"])

        goals_home_obs = x_train_cop["FTHG"].to_numpy()
        goals_away_obs = x_train_cop["FTAG"].to_numpy()
        home_team = x_train_cop["HomeTeamId"].to_numpy()
        away_team = x_train_cop["AwayTeamId"].to_numpy()
        self.trace = self.hierarchical_bayes(goals_home_obs, goals_away_obs, home_team, away_team)

    def predict(self, home_team: str, away_team: str) -> GoalMatrix:
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

        home_theta = np.exp(
            intercept + home[home_team_id] + atts[home_team_id] + defs[away_team_id]
        )
        away_theta = np.exp(intercept + atts[away_team_id] + defs[home_team_id])

        return home_theta, away_theta

    def get_samples(self, home_team: str, away_team: str) -> tuple[np.ndarray, np.ndarray]:
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

        team_id = self.label.transform([home_team, away_team])
        posterior = self.trace.posterior
        home_team_id = team_id[0]
        away_team_id = team_id[1]
        home = posterior["home"].values
        intercept = posterior["intercept"].values
        atts = posterior["atts"].values
        defs = posterior["defs"].values
        lambda_h = np.exp(
            intercept + home[..., home_team_id] + atts[..., home_team_id] + defs[..., away_team_id]
        )
        lambda_a = np.exp(intercept + atts[..., away_team_id] + defs[..., home_team_id])
        return lambda_h.flatten(), lambda_a.flatten()

    def hierarchical_bayes(
        self,
        goals_home_obs: np.ndarray,
        goals_away_obs: np.ndarray,
        home_team: np.ndarray,
        away_team: np.ndarray,
    ):
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

            # Goal likelihood
            pm.Poisson("home_goals", mu=home_theta, observed=goals_home_data)
            pm.Poisson("away_goals", mu=away_theta, observed=goals_away_data)
            # Sample with improved settings
            trace = pm.sample(
                2000, tune=500, cores=6, target_accept=0.95, return_inferencedata=True, nuts_sampler="numpyro")
        return trace


class LogNormalBayesian(ProtoPoisson):
    def __init__(self, n_teams: int, n_goals: int):
        self.n_teams = n_teams
        self.n_goals = n_goals
        self.label = preprocessing.LabelEncoder()

    @verify_required_column(column_names={"HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG", "Date", "Time"})
    def fit(self, X_train: pd.DataFrame):
        x_train_cop = copy(X_train)
        self.label.fit(X_train["HomeTeam"])  # type: ignore
        x_train_cop["HomeTeamId"] = self.label.transform(X_train["HomeTeam"])
        x_train_cop["AwayTeamId"] = self.label.transform(X_train["AwayTeam"])

        goals_home_obs = x_train_cop["FTHG"].to_numpy()
        goals_away_obs = x_train_cop["FTAG"].to_numpy()
        home_team = x_train_cop["HomeTeamId"].to_numpy()
        away_team = x_train_cop["AwayTeamId"].to_numpy()
        week_idx = process_game_week_index(df=X_train)
        self.trace = self.hierarchical_bayes_dyn_PL(goals_home_obs=goals_home_obs, goals_away_obs=goals_away_obs, 
                                                    home_team=home_team, away_team=away_team,week_idx=week_idx)

    def predict(self, home_team: str, away_team: str) -> GoalMatrix:
        team_id = self.label.transform([home_team, away_team])

        home_goal_expectation, away_goal_expectation = self.goal_expectation(
            home_team_id=team_id[0], away_team_id=team_id[1], point_estimate=True
        )

        home_probs = stats.poisson.pmf(range(self.n_goals), home_goal_expectation)
        away_probs = stats.poisson.pmf(range(self.n_goals), away_goal_expectation)

        goals_matrix = GoalMatrix(home_probs, away_probs)
        return goals_matrix
    
    def goal_expectation(
        self,
        home_team_id: int,
        away_team_id: int,
        point_estimate: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return λ-parameters for a single fixture.

        Parameters
        ----------
        home_team_id, away_team_id : int
            Encoded team IDs.
        week_index : int | None
            Which match-week to evaluate (0-based).  None = use the last one in
            the trace.
        point_estimate : bool
            True  → posterior means (floats)  
            False → full vector of samples (1-D).

        Returns
        -------
        λ_home , λ_away : float | np.ndarray
        """
        pst        = self.trace.posterior          # shorthand
        attack     = pst["attack"].values          # (chains, draws, teams, weeks)
        defence    = pst["defence"].values
        home_adv   = pst["home_adv_rw"].values     # (chains, draws, weeks)
        intercept  = pst["intercept"].values       # (chains, draws)

        week_index = attack.shape[-1] - 1      # last week in the array

        # slice once to keep everything vectorised
        λ_home = np.exp(
            intercept
            + home_adv[..., week_index]
            + attack[..., home_team_id, week_index]
            - defence[..., away_team_id, week_index]
        )

        λ_away = np.exp(
            intercept
            + attack[..., away_team_id, week_index]
            - defence[..., home_team_id, week_index]
        )

        if point_estimate:
            return λ_home.mean().item(), λ_away.mean().item()

        return λ_home.ravel(), λ_away.ravel()

    # -------------------------------------------------------------------------
    # 2.  Posterior-draw sampler for a fixture
    # -------------------------------------------------------------------------
    def get_samples(
        self,
        home_team: str,
        away_team: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Posterior predictive λ-samples for *home_team vs away_team*.

        Returns
        -------
        λ_home_samples , λ_away_samples : 1-D np.ndarrays (n_samples ~= n_chains×n_draws)
        """
        home_id, away_id = self.label.transform([home_team, away_team])
        return self.goal_expectation(
            home_team_id=home_id,
            away_team_id=away_id,
            point_estimate=False,
        )


    def hierarchical_bayes_dyn_PL(
        self,
        goals_home_obs: np.ndarray,
        goals_away_obs: np.ndarray,
        home_team: np.ndarray,
        away_team: np.ndarray,
        week_idx: np.ndarray,
        cores: int | None = None,
    ):
        """Dynamic Poisson‑lognormal football model in PyMC (PyTensor backend)."""

        n_matches = goals_home_obs.shape[0]
        n_weeks = int(week_idx.max()) + 1

        with pm.Model() as model:
            # ----------------------------------------------------------------------------
            # 1.   Observed data holders (allows later posterior predictive updates)
            # ----------------------------------------------------------------------------
            goals_home_d = pm.Data("goals_home", goals_home_obs)
            goals_away_d = pm.Data("goals_away", goals_away_obs)
            home_team_d  = pm.Data("home_team", home_team)
            away_team_d  = pm.Data("away_team", away_team)
            week_d       = pm.Data("week", week_idx)

            # ----------------------------------------------------------------------------
            # 2.   Hyper‑priors
            # ----------------------------------------------------------------------------
            sigma_att = pm.Exponential("sigma_att", 1.0)
            sigma_def = pm.Exponential("sigma_def", 1.0)
            sigma_home = pm.Exponential("sigma_home_adv", 1.0)
            sigma_obs = pm.Exponential("sigma_obs", 1.0)  # over‑dispersion for PLN


            # ----------------------------------------------------------------------------
            # 3.   Dynamic parameters: Gaussian random walks per team
            # ----------------------------------------------------------------------------
            attack = pm.GaussianRandomWalk(
                "attack", sigma=sigma_att, shape=(self.n_teams, n_weeks)
            )
            defence = pm.GaussianRandomWalk(
                "defence", sigma=sigma_def, shape=(self.n_teams, n_weeks)
            )
            home_adv_rw = pm.GaussianRandomWalk(
                "home_adv_rw", sigma=sigma_home, shape=n_weeks
            )

            intercept = pm.Normal("intercept", 0.0, 1.0)

            # ----------------------------------------------------------------------------
            # 4.   Linear predictor for each match (log‑rate)
            # ----------------------------------------------------------------------------
            theta_home = (
                intercept
                + home_adv_rw[week_d]
                + attack[home_team_d, week_d]
                - defence[away_team_d, week_d]
            )
            theta_away = (
                intercept
                + attack[away_team_d, week_d]
                - defence[home_team_d, week_d]
            )

            # ----------------------------------------------------------------------------
            # 5.   Poisson‑lognormal likelihood (additive Normal noise on log‑rate)
            # ----------------------------------------------------------------------------
            eps_home = pm.Normal("eps_home", mu=0.0, sigma=sigma_obs, shape=n_matches)
            eps_away = pm.Normal("eps_away", mu=0.0, sigma=sigma_obs, shape=n_matches)

            rate_home = pm.math.exp(theta_home + eps_home)
            rate_away = pm.math.exp(theta_away + eps_away)

            pm.Poisson("home_goals", mu=rate_home, observed=goals_home_d)
            pm.Poisson("away_goals", mu=rate_away, observed=goals_away_d)

            # ----------------------------------------------------------------------------
            # 6.   Inference (NUTS via NumPyro backend for GPU‑speed on Jetson)
            # ----------------------------------------------------------------------------
            trace = pm.sample(
                2000,
                tune=500,
                target_accept=0.95,
                nuts_sampler="numpyro",
                cores=cores,
                return_inferencedata=True,
            )

        return trace