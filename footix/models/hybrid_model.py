from copy import copy
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import scipy.stats as stats
from sklearn import preprocessing

from footix.implied_odds import shin
from footix.models.score_matrix import GoalMatrix
from footix.models.utils import implicit_intensities
from footix.utils.decorators import verify_required_column
from footix.utils.typing import ProtoBayes


class MixtureBayesian(ProtoBayes):
    def __init__(self, n_teams: int, n_goals: int):
        self.n_teams = n_teams
        self.n_goals = n_goals
        self.label = preprocessing.LabelEncoder()

    @verify_required_column(
        column_names={
            "home_team",
            "away_team",
            "ftr",
            "fthg",
            "ftag",
            "b365_h",
            "b365_d",
            "b365_a",
        }
    )
    def fit(self, X_train: pd.DataFrame):
        x_train_cop = copy(X_train)
        self.label.fit(X_train["home_team"])  # type: ignore
        x_train_cop["home_team_id"] = self.label.transform(X_train["home_team"])
        x_train_cop["away_team_id"] = self.label.transform(X_train["away_team"])

        goals_home_obs = x_train_cop["fthg"].to_numpy()
        goals_away_obs = x_train_cop["ftag"].to_numpy()
        home_team = x_train_cop["home_team_id"].to_numpy()
        away_team = x_train_cop["away_team_id"].to_numpy()

        tmp_list_odds = x_train_cop[["b365_h", "b365_d", "b365_a"]].to_numpy()
        proba_shin_array = self.compute_implied_odds(list_odds=tmp_list_odds)
        thetas_intensities = implicit_intensities(proba_from_odds=proba_shin_array, max_iter=1000)

        self.trace = self.hierarchical_bayes(
            goals_home_obs,
            goals_away_obs,
            home_team,
            away_team,
            theta_hat_home=thetas_intensities[:, 0],
            theta_hat_away=thetas_intensities[:, 1],
        )

    def compute_implied_odds(self, list_odds: list[list[float]] | np.ndarray) -> np.ndarray:
        """Compute the implied odds from the given list of odds.

        Parameters
        ----------
        list_odds : list or np.ndarray
            List of odds [odds_home, odds_draw, odds_away].

        Returns
        -------
        np.ndarray
            Implied odds.

        """
        tmp_proba = []
        for odd in list_odds:
            tmp_shin, _ = shin(odd)
            tmp_proba.append(tmp_shin)
        return np.asarray(tmp_proba)

    def predict(self, home_team: str, away_team: str, **kwargs: Any) -> GoalMatrix:
        bookmaker_odds: list[float] | None = kwargs.get("bookmaker_odds")  # type: ignore
        if bookmaker_odds is None:
            raise ValueError("bookmaker_odds is not defined")
        team_id = self.label.transform([home_team, away_team])

        home_goal_expectation, away_goal_expectation = self.goal_expectation(
            home_team_id=team_id[0], away_team_id=team_id[1], bookmaker_odds=bookmaker_odds
        )

        home_probs = stats.poisson.pmf(range(self.n_goals), home_goal_expectation)
        away_probs = stats.poisson.pmf(range(self.n_goals), away_goal_expectation)

        goals_matrix = GoalMatrix(home_probs, away_probs)
        return goals_matrix

    def goal_expectation(self, home_team_id: int, away_team_id: int, bookmaker_odds: list[float]):
        proba_shin, _ = shin(bookmaker_odds)
        proba_shin = np.asarray([proba_shin])
        theta_intensities = implicit_intensities(proba_shin)[0]

        posterior = self.trace.posterior
        home = posterior["home"].mean(dim=["chain", "draw"]).values
        intercept = posterior["intercept"].mean(dim=["chain", "draw"]).values
        atts = posterior["atts"].mean(dim=["chain", "draw"]).values
        defs = posterior["defs"].mean(dim=["chain", "draw"]).values
        w = posterior["w"].mean(dim=["chain", "draw"]).values

        home_theta = (
            w * np.exp(intercept + home[home_team_id] + atts[home_team_id] + defs[away_team_id])
            + (1 - w) * theta_intensities[0]
        )
        away_theta = (
            w * np.exp(intercept + atts[away_team_id] + defs[home_team_id])
            + (1 - w) * theta_intensities[1]
        )

        return home_theta, away_theta

    def get_samples(
        self, home_team: str, away_team: str, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
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
        bookmaker_odds: list[float] | None = kwargs.get("bookmaker_odds")  # type: ignore
        if bookmaker_odds is None:
            raise ValueError("bookmaker_odds is not defined")
        team_id = self.label.transform([home_team, away_team])
        proba_shin, _ = shin(bookmaker_odds)
        proba_shin = np.asarray([proba_shin])
        theta_intensities = implicit_intensities(proba_shin)[0]
        posterior = self.trace.posterior
        home_team_id = team_id[0]
        away_team_id = team_id[1]
        home = posterior["home"].values
        intercept = posterior["intercept"].values
        w = posterior["w"].values
        atts = posterior["atts"].values
        defs = posterior["defs"].values

        lambda_h = (
            w
            * np.exp(
                intercept
                + home[..., home_team_id]
                + atts[..., home_team_id]
                + defs[..., away_team_id]
            )
            + (1 - w) * theta_intensities[0]
        )
        lambda_a = (
            w * np.exp(intercept + atts[..., away_team_id] + defs[..., home_team_id])
            + (1 - w) * theta_intensities[1]
        )
        return lambda_h.flatten(), lambda_a.flatten()

    def hierarchical_bayes(
        self,
        goals_home_obs: np.ndarray,
        goals_away_obs: np.ndarray,
        home_team: np.ndarray,
        away_team: np.ndarray,
        theta_hat_home: np.ndarray,
        theta_hat_away: np.ndarray,
    ):
        with pm.Model():
            goals_home_data = pm.Data("goals_home", goals_home_obs)
            goals_away_data = pm.Data("goals_away", goals_away_obs)
            home_team_data = pm.Data("home_team", home_team)
            away_team_data = pm.Data("away_team", away_team)
            theta_hat_home_data = pm.Data("theta_hat_home", theta_hat_home)
            theta_hat_away_data = pm.Data("theta_hat_away", theta_hat_away)

            intercept = pm.Normal("intercept", mu=3, sigma=1)
            home_adv = pm.Normal("home", mu=0, sigma=1, shape=self.n_teams)

            # Attack / defence with non‑centred shrinkage
            tau_att = pm.HalfNormal("tau_att", sigma=2)
            tau_def = pm.HalfNormal("tau_def", sigma=2)
            raw_atts = pm.Normal("raw_atts", mu=0, sigma=1, shape=self.n_teams)
            raw_defs = pm.Normal("raw_defs", mu=0, sigma=1, shape=self.n_teams)
            atts = pm.Deterministic(
                "atts", (raw_atts * tau_att) - pm.math.mean(raw_atts * tau_att)
            )
            defs = pm.Deterministic(
                "defs", (raw_defs * tau_def) - pm.math.mean(raw_defs * tau_def)
            )

            theta_hist_home = pm.math.exp(
                intercept + home_adv[home_team_data] + atts[home_team_data] + defs[away_team_data]
            )
            theta_hist_away = pm.math.exp(intercept + atts[away_team_data] + defs[home_team_data])

            w = pm.Beta("w", alpha=2, beta=2)
            rate_home = w * theta_hist_home + (1 - w) * theta_hat_home_data
            rate_away = w * theta_hist_away + (1 - w) * theta_hat_away_data

            pm.Poisson("home_goals", mu=rate_home, observed=goals_home_data)
            pm.Poisson("away_goals", mu=rate_away, observed=goals_away_data)

            trace = pm.sample(
                2000,
                tune=500,
                target_accept=0.95,
                return_inferencedata=True,
                cores=6,
                nuts_sampler="numpyro",
            )
            return trace
