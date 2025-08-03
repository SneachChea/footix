import logging
import os
import warnings
from copy import copy
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import scipy.stats as stats
from sklearn import preprocessing

from footix.models.score_matrix import GoalMatrix
from footix.utils.decorators import verify_required_column
from footix.utils.typing import SampleProbaResult

logger = logging.getLogger(name=__name__)


class DixonColesBayesian:
    """Version PyMC du modèle Dixon-Coles (ajustement low-score)."""

    def __init__(self, n_teams: int, n_goals: int):
        self.n_teams = n_teams
        self.n_goals = n_goals
        self.label = preprocessing.LabelEncoder()

    # ---------------------------------------------------------------------
    #  Entraînement
    # ---------------------------------------------------------------------
    @verify_required_column(column_names={"home_team", "away_team", "fthg", "ftag"})
    def fit(self, X_train: pd.DataFrame):
        x = copy(X_train)
        self.label.fit(x["home_team"])

        x["home_id"] = self.label.transform(x["home_team"])
        x["away_id"] = self.label.transform(x["away_team"])

        goals_home_obs = x["fthg"].to_numpy()
        goals_away_obs = x["ftag"].to_numpy()
        home_id = x["home_id"].to_numpy().astype("int64")
        away_id = x["away_id"].to_numpy().astype("int64")

        # Trace = échantillons postérieurs
        self.trace = self._hierarchical_bayes(goals_home_obs, goals_away_obs, home_id, away_id)

    # ---------------------------------------------------------------------
    #  Prédiction : marges indépendantes (même API que ton modèle NB)
    # ---------------------------------------------------------------------
    def predict(self, home_team: str, away_team: str, **kwargs: Any) -> GoalMatrix:
        if kwargs:
            warnings.warn(
                f"Ignoring unexpected keyword arguments: {list(kwargs.keys())}",
                stacklevel=2,
            )

        home_id, away_id = self.label.transform([home_team, away_team])

        mu_home, mu_away = self.goal_expectation(home_team_id=home_id, away_team_id=away_id)

        ks = np.arange(self.n_goals)
        home_probs = stats.poisson.pmf(ks, mu_home)
        away_probs = stats.poisson.pmf(ks, mu_away)
        rho = self.trace.posterior["low_score_corr"].mean(("chain", "draw")).values.item()
        corr_matrix = np.ones((self.n_goals, self.n_goals), dtype=float)

        corr_matrix[1, 1] = 1 - rho
        corr_matrix[0, 1] = 1 + rho * mu_home
        corr_matrix[1, 0] = 1 + rho * mu_away
        corr_matrix[0, 0] = 1 - rho * mu_home * mu_away

        return GoalMatrix(home_probs, away_probs, correlation_matrix=corr_matrix)

    # ---------------------------------------------------------------------
    #  Espérance des buts (moyenne de la postérieure)
    # ---------------------------------------------------------------------
    def goal_expectation(self, home_team_id: int, away_team_id: int):
        post = self.trace.posterior

        intercept = post["intercept"].mean(("chain", "draw")).values
        home_adv_coef = post["home_adv"].mean(("chain", "draw")).values
        attack = post["attack"].mean(("chain", "draw")).values
        defense = post["defense"].mean(("chain", "draw")).values

        mu_home = np.exp(intercept + home_adv_coef + attack[home_team_id] + defense[away_team_id])
        mu_away = np.exp(intercept + attack[away_team_id] + defense[home_team_id])
        return mu_home, mu_away

    @cache
    def get_samples(self, home_team: str, away_team: str, **kwargs: Any) -> SampleProbaResult:
        if kwargs:
            warnings.warn(
                f"Ignoring unexpected keyword arguments: {list(kwargs.keys())}",
                stacklevel=2,
            )

        home_id, away_id = self.label.transform([home_team, away_team])

        post = self.trace.posterior

        intercept = post["intercept"].stack(sample=("chain", "draw")).values
        home_adv = self.trace.posterior["home_adv"].stack(sample=("chain", "draw")).values
        attack = self.trace.posterior["attack"].stack(sample=("chain", "draw")).values
        defense = self.trace.posterior["defense"].stack(sample=("chain", "draw")).values
        rho = self.trace.posterior["low_score_corr"].stack(sample=("chain", "draw")).values

        n_samples = intercept.shape[0]

        prob_H_list = []
        prob_D_list = []
        prob_A_list = []

        def dc_tau(scores_h, scores_a, lam_h, lam_a, rho):
            tau = np.ones_like(scores_h, dtype=np.float64)
            m00 = (scores_h == 0) & (scores_a == 0)
            m01 = (scores_h == 0) & (scores_a == 1)
            m10 = (scores_h == 1) & (scores_a == 0)
            m11 = (scores_h == 1) & (scores_a == 1)
            tau[m00] = 1.0 - lam_h * lam_a * rho
            tau[m01] = 1.0 + lam_h * rho
            tau[m10] = 1.0 + lam_a * rho
            tau[m11] = 1.0 - rho
            return tau

        for i in range(n_samples):
            lam_home = np.exp(
                intercept[i] + home_adv[home_id] + attack[home_id, i] + defense[away_id, i]
            )
            lam_away = np.exp(intercept[i] + attack[away_id, i] + defense[home_id, i])
            sh = np.random.poisson(lam_home, size=200)
            sa = np.random.poisson(lam_away, size=200)
            tau = dc_tau(sh, sa, lam_home, lam_away, rho[i])
            w = tau
            w_sum = w.sum()
            mH = sh > sa
            mD = sh == sa
            mA = sh < sa

            prob_H_list.append(w[mH].sum() / w_sum if mH.any() else 0.0)
            prob_D_list.append(w[mD].sum() / w_sum if mD.any() else 0.0)
            prob_A_list.append(w[mA].sum() / w_sum if mA.any() else 0.0)

        return SampleProbaResult(
            proba_home=np.asarray(prob_H_list),
            proba_draw=np.asarray(prob_D_list),
            proba_away=np.asarray(prob_A_list),
        )

    # ---------------------------------------------------------------------
    #  Construction du modèle hiérarchique + échantillonnage
    # ---------------------------------------------------------------------
    def _hierarchical_bayes(
        self,
        goals_home_obs: np.ndarray,
        goals_away_obs: np.ndarray,
        home_id: np.ndarray,
        away_id: np.ndarray,
    ):
        with pm.Model():
            # ---------------------------
            #  Données observées
            # ---------------------------
            gh = pm.Data("gh", goals_home_obs)
            ga = pm.Data("ga", goals_away_obs)
            h_id = pm.Data("h_id", home_id)
            a_id = pm.Data("a_id", away_id)

            # ---------------------------
            #  Effets hiérarchiques
            # ---------------------------
            intercept = pm.Normal("intercept", mu=0.9, sigma=1.0)

            # Home advantage (un seul paramètre ; somme-à-zéro inutile)
            home_adv = pm.Normal("home_adv", mu=0, sigma=0.5)

            # Attaques / défenses non centrées
            tau_att = pm.HalfNormal("tau_att", sigma=1.5)
            raw_att = pm.Normal("raw_att", mu=0, sigma=1, shape=self.n_teams)
            attack = pm.Deterministic(
                "attack", (raw_att * tau_att) - pm.math.mean(raw_att * tau_att)
            )

            tau_def = pm.HalfNormal("tau_def", sigma=1.5)
            raw_def = pm.Normal("raw_def", mu=0, sigma=1, shape=self.n_teams)
            defense = pm.Deterministic(
                "defense", (raw_def * tau_def) - pm.math.mean(raw_def * tau_def)
            )

            # Paramètre Dixon-Coles : corrélation des faibles scores
            low_score_corr = pm.Uniform("low_score_corr", lower=0.0, upper=1.0)

            # ---------------------------
            #  Taux Poisson attendus
            # ---------------------------
            rate_home = pm.math.exp(intercept + home_adv + attack[h_id] + defense[a_id])
            rate_away = pm.math.exp(intercept + attack[a_id] + defense[h_id])

            def dc_logp(goals_h, goals_a, lam_h, lam_a, rho):
                base = pm.logp(pm.Poisson.dist(mu=lam_h), goals_h) + pm.logp(
                    pm.Poisson.dist(mu=lam_a), goals_a
                )

                # Conditions (avec pt.eq)
                zero_zero = pt.eq(goals_h, 0) & pt.eq(goals_a, 0)
                one_one = pt.eq(goals_h, 1) & pt.eq(goals_a, 1)
                one_zero = pt.eq(goals_h, 1) & pt.eq(goals_a, 0)
                zero_one = pt.eq(goals_h, 0) & pt.eq(goals_a, 1)

                # Initialisation de tau
                tau = pt.ones_like(goals_h, dtype="float64")

                # Ajouts conditionnels (avec pt.switch)
                tau = pt.switch(
                    one_one,
                    1 - rho,
                    tau,
                )
                tau = pt.switch(
                    zero_zero,
                    1 + rho * lam_h * lam_a,
                    tau,
                )
                tau = pt.switch(
                    one_zero,
                    1 + rho * lam_a,
                    tau,
                )
                tau = pt.switch(
                    zero_one,
                    1 + rho * lam_h,
                    tau,
                )

                return base + pt.log(tau)

            # Ajout au log-posterior
            pm.Potential("dc_like", dc_logp(gh, ga, rate_home, rate_away, low_score_corr).sum())

            trace = pm.sample(
                2000,
                tune=1000,
                cores=os.cpu_count(),
                target_accept=0.95,
                nuts_sampler="numpyro",  # rapide + robustesse
                init="adapt_diag_grad",
                return_inferencedata=True,
            )
        return trace
