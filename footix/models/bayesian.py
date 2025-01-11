from copy import copy

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import scipy.stats as stats
from sklearn import preprocessing

from footix.models.protocol_model import ProtoPoisson
from footix.models.score_matrix import GoalMatrix
from footix.utils.decorators import verify_required_column


class Bayesian(ProtoPoisson):
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
        # get parameters

        home = np.mean(self.trace["home"])
        intercept = np.mean(self.trace["intercept"])
        atts_home = np.mean([x[home_team_id] for x in self.trace["atts"]])
        atts_away = np.mean([x[away_team_id] for x in self.trace["atts"]])
        defs_home = np.mean([x[home_team_id] for x in self.trace["defs"]])
        defs_away = np.mean([x[away_team_id] for x in self.trace["defs"]])

        # calculate theta
        home_theta = np.exp(intercept + home + atts_home - defs_away)
        away_theta = np.exp(intercept + atts_away - defs_home)

        # return the average per team
        return home_theta, away_theta

    def hierarchical_bayes(
        self,
        goals_home_obs: np.ndarray,
        goals_away_obs: np.ndarray,
        home_team: np.ndarray,
        away_team: np.ndarray,
    ):
        with pm.Model():
            # Home advantage

            grp_att = pm.Categorical("grp_att", p=[0.5, 0.5], shape=self.n_teams)
            grp_def = pm.Categorical("grp_def", p=[0.5, 0.5], shape=self.n_teams)
            # Home advantage and intercept
            home = pm.Normal("home", mu=0, sigma=1)
            intercept = pm.Normal("intercept", mu=3, sigma=1)
            # Group-level priors
            sigma_att = pm.Gamma("sigma_att", alpha=0.1, beta=0.1, shape=2)
            sigma_def = pm.Gamma("sigma_def", alpha=0.1, beta=0.1, shape=2)

            # Team-specific attack and defense effects
            attack = pm.Normal("attack", mu=0, sigma=sigma_att[grp_att], shape=self.n_teams)
            defense = pm.Normal("defense", mu=0, sigma=sigma_def[grp_def], shape=self.n_teams)

            # Sum-zero constraints
            atts = pm.Deterministic("atts", attack - pt.mean(attack))
            defs = pm.Deterministic("defs", defense - pt.mean(defense))

            # Calculate theta
            home_theta = pt.exp(intercept + home + atts[home_team] - defs[away_team])
            away_theta = pt.exp(intercept + atts[away_team] - defs[home_team])

            # Goal expectation
            pm.Poisson("home_goals", mu=home_theta, observed=goals_home_obs)
            pm.Poisson("away_goals", mu=away_theta, observed=goals_away_obs)

            return pm.sample(3000, tune=2000, chains=4, cores=6, return_inferencedata=False)
