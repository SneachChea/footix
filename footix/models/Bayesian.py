import pymc as pm
import numpy as np
import pytensor.tensor as pt
import pandas as pd
import matplotlib.pyplot as plt
import footix as ft
from collections import defaultdict
import scipy.stats as stats
from .abstract_model import CustomModel
from typing import Any, List, Dict, Tuple, Union
from sklearn.preprocessing import LabelEncoder
from copy import copy
from footix.utils import DICO_COMPATIBILITY

class Bayesian(CustomModel):
    def __init__(self, n_teams: int):
        super().__init__(n_teams)

    def fit(self, X_train: pd.DataFrame):
        if hasattr(self, "model"):
            raise ValueError("Object already fitted")
        x_train_cop = copy(X_train)
        if not hasattr(self, "label"):
            self.label = LabelEncoder()
            self.label.fit(X_train["HomeTeam"])
        x_train_cop["HomeTeamId"] = self.label.transform(X_train["HomeTeam"])
        x_train_cop["AwayTeamId"] = self.label.transform(X_train["AwayTeam"])

        goals_home_obs = x_train_cop["FTHG"].values
        goals_away_obs = x_train_cop["FTAG"].values
        home_team = x_train_cop["HomeTeamId"].values
        away_team = x_train_cop["AwayTeamId"].values
        self.model = self.modelPoisson(goals_home_obs, goals_away_obs, home_team, away_team)
        with self.model:
            self.trace = pm.sample(2000, tune=1000, cores=6, return_inferencedata=False)
        
    def predict(self, HomeTeam: str, AwayTeam: str, score_matrix: bool = False)-> Union[Tuple[float, np.ndarray], Tuple]:
        if not hasattr(self, "model"):
            raise AttributeError("Model is not trained. Please train it.")
        home_team = DICO_COMPATIBILITY[HomeTeam]
        away_team = DICO_COMPATIBILITY[AwayTeam]

        home_team_id = self.label.transform([home_team])
        away_team_id = self.label.transform([away_team])

        home_goal_expectation, away_goal_expectation = self.goal_expectation(home_team_id = home_team_id, away_team_id=away_team_id)

        home_probs = stats.poisson.pmf(range(6), home_goal_expectation)
        away_probs = stats.poisson.pmf(range(6), away_goal_expectation)

        m = np.outer(home_probs, away_probs)
        home = np.sum(np.tril(m, -1))
        draw = np.sum(np.diag(m))
        away = np.sum(np.triu(m, 1))

        if score_matrix:
            return (home, draw, away), m
        return home, draw, away

    def goal_expectation(self, home_team_id, away_team_id):   
        # get parameters
        home = np.mean(self.trace["home"])
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


    def modelPoisson(self, goals_home_obs: np.ndarray, goals_away_obs: np.ndarray, home_team: np.ndarray, away_team: np.ndarray):
        with pm.Model() as model:
            # home advantage
            home = pm.Flat("home")
            intercept = pm.Normal('intercept', mu=3, sigma=1)
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
            home_theta = pt.exp(intercept + home + atts[home_team] + defs[away_team])
            away_theta = pt.exp(intercept + atts[away_team] + defs[home_team])

            # goal expectation
            home_points = pm.Poisson("home_goals", mu=home_theta, observed=goals_home_obs)
            away_points = pm.Poisson("away_goals", mu=away_theta, observed=goals_away_obs)
            return model