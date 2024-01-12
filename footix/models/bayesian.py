from copy import copy
from typing import TypeVar
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import scipy.stats as stats

from sklearn import preprocessing

from footix.models.scored_matrix import GoalMatrix
from footix.utils.decorators import verify_required_column
from footix.utils.utils import DICO_COMPATIBILITY

T = TypeVar("MultiTrace")

class Bayesian:
    def __init__(self, n_teams: int):
        self.n_teams = n_teams

    @verify_required_column(column_names={"HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"})
    def fit(self, x_train: pd.DataFrame):
        x_train_cop = copy(x_train)
        if not hasattr(self, "label"):
            self.label = preprocessing.LabelEncoder()
            self.label.fit(x_train["HomeTeam"]) # type: ignore
        x_train_cop["HomeTeamId"] = self.label.transform(x_train["HomeTeam"])
        x_train_cop["AwayTeamId"] = self.label.transform(x_train["AwayTeam"])

        goals_home_obs = x_train_cop["FTHG"].to_numpy()
        goals_away_obs = x_train_cop["FTAG"].to_numpy()
        home_team = x_train_cop["HomeTeamId"].to_numpy()
        away_team = x_train_cop["AwayTeamId"].to_numpy()
        self.trace = self.hierarchical_bayes(goals_home_obs, goals_away_obs, home_team, away_team)

    def predict(
        self,
        input_home_team: str,
        input_away_team: str,
        score_matrix: bool = False,
        cote_fdj: bool = True,
    ) -> tuple[float, np.ndarray] | tuple:
        if cote_fdj:
            home_team = DICO_COMPATIBILITY[input_home_team]
            away_team = DICO_COMPATIBILITY[input_away_team]
        else:
            home_team = input_home_team
            away_team = input_away_team

        team_id = self.label.transform([home_team, away_team])

        home_goal_expectation, away_goal_expectation = self.goal_expectation(
            home_team_id=team_id[0], away_team_id=team_id[1]
        )

        home_probs = stats.poisson.pmf(range(6), home_goal_expectation)
        away_probs = stats.poisson.pmf(range(6), away_goal_expectation)

        goals_matrix = GoalMatrix(home_probs, away_probs)
        home, draw, away = goals_matrix.return_probas()

        if score_matrix:
            return (home, draw, away), goals_matrix
        return home, draw, away

    def goal_expectation(self, home_team_id: int, away_team_id: int):
        # get parameters
        home = np.mean([x[home_team_id] for x in self.trace["home"]])
        intercept = np.mean(self.trace["intercept"])
        atts_home = np.mean([x[home_team_id] for x in self.trace["atts"]])
        atts_away = np.mean([x[away_team_id] for x in self.trace["atts"]])
        defs_home = np.mean([x[home_team_id] for x in self.trace["defs"]])
        defs_away = np.mean([x[away_team_id] for x in self.trace["defs"]])

        # calculate theta
        home_theta = np.exp(intercept + home + atts_home + defs_away)
        away_theta = np.exp(intercept + atts_away + defs_home)

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
            home = pm.Normal("home", mu=0, sigma=1, shape=self.n_teams)
            intercept = pm.Normal("intercept", mu=3, sigma=1)
            # attack ratings
            tau_att = pm.HalfNormal("tau_att", sigma=2)
            atts_star = pm.Normal("atts_star", mu=0, tau=tau_att, shape=self.n_teams)

            # defence ratings
            tau_def = pm.HalfNormal("tau_def", sigma=2)
            def_star = pm.Normal("def_star", mu=0, tau=tau_def, shape=self.n_teams)

            # apply sum zero constraints
            atts = pm.Deterministic("atts", atts_star - pt.mean(atts_star))
            defs = pm.Deterministic("defs", def_star - pt.mean(def_star))

            # calulate theta
            home_theta = pt.exp(intercept + home[home_team] + atts[home_team] + defs[away_team])
            away_theta = pt.exp(intercept + atts[away_team] + defs[home_team])

            # goal expectation
            pm.Poisson("home_goals", mu=home_theta, observed=goals_home_obs)
            pm.Poisson("away_goals", mu=away_theta, observed=goals_away_obs)

            return pm.sample(2000, tune=500, cores=6, return_inferencedata=False)