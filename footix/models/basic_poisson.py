import logging

import numpy as np
import pandas as pd
import scipy.optimize as optimize
import scipy.stats as stats

import footix.models.score_matrix as score_matrix
import footix.models.utils as model_utils

logger = logging.getLogger(name=__name__)


class BasicPoisson:
    def __init__(self, n_teams: int, n_goals: int) -> None:
        self.n_teams = n_teams
        self.n_goals = n_goals

    def fit(self, X_train: pd.DataFrame) -> None:
        self.dict_teams = self.mapping_team_index(X_train["HomeTeam"])
        self._sanity_check(X_train["AwayTeam"])
        goals_home, basis_home = model_utils.compute_goals_home_vectors(
            X_train, map_teams=self.dict_teams, nbr_team=self.n_teams
        )
        goals_away, basis_away = model_utils.compute_goals_away_vectors(
            X_train, map_teams=self.dict_teams, nbr_team=self.n_teams
        )
        optimization_result = optimize.minimize(
            np_poisson_likelihood,
            x0=np.zeros(41),
            args=(goals_home, goals_away, basis_home, basis_away),
        )
        if not optimization_result.success:
            logger.warning("Minimization routine was not successful.")
        self.model_params = optimization_result.x

    def print_parameters(self) -> None:
        gamma = self.model_params[0]
        alphas = self.model_params[1:21]
        betas = self.model_params[21:]
        str_resume = f"Gamma = {gamma} \n"
        str_alpha = "".join(
            [f"alpha team-{team}= {alphas[idx]} \n" for team, idx in self.dict_teams.items()]
        )
        str_beta = "".join(
            [f"beta team-{team}= {betas[idx]} \n" for team, idx in self.dict_teams.items()]
        )
        print(str_resume + str_alpha + str_beta)

    def predict(self, home_team: str, away_team: str) -> score_matrix.GoalMatrix:
        if home_team not in self.dict_teams.keys():
            raise ValueError(f"Home team {home_team} is not in the list.")
        if away_team not in self.dict_teams.keys():
            raise ValueError(f"Away team {away_team} is not in the list.")
        gamma = self.model_params[0]
        alphas = self.model_params[1:21]
        betas = self.model_params[21:]
        i = self.dict_teams[home_team]
        j = self.dict_teams[away_team]
        lamb = np.exp(alphas[i] + betas[j] + gamma)
        mu = np.exp(alphas[j] + betas[i])
        return score_matrix.GoalMatrix(
            home_probs=poisson_proba(lambda_params=lamb, k=self.n_goals),
            away_probs=poisson_proba(lambda_params=mu, k=self.n_goals),
        )

    def mapping_team_index(self, teams: pd.Series) -> dict[str, int]:
        list_teams = list(sorted(teams.unique()))
        return {element: index for index, element in enumerate(list_teams)}

    def _sanity_check(self, teams: pd.Series) -> None:
        dict_teams_away = self.mapping_team_index(teams)
        if self.dict_teams != dict_teams_away:
            raise ValueError(
                "Not every teams have played at home and away. Please give another dataset."
            )
        if len(self.dict_teams) != self.n_teams:
            raise ValueError(f"Expecting {self.n_teams} teams, only got {len(self.dict_teams)}.")


def np_poisson_likelihood(
    params: np.ndarray,
    goals_home: np.ndarray,
    goals_away: np.ndarray,
    basis_home: np.ndarray,
    basis_away: np.ndarray,
) -> float:
    gamma = params[0]
    alphas = params[1:21]
    betas = params[21:]
    log_lamdas = np.dot(basis_home, alphas) + np.dot(basis_away, betas) + gamma
    log_mus = np.dot(basis_away, alphas) + np.dot(basis_home, betas)
    lambdas = np.exp(log_lamdas)
    mus = np.exp(log_mus)
    log = lambdas + mus - goals_home * log_lamdas - goals_away * log_mus
    return np.sum(log)


def poisson_proba(lambda_params: float, k: int) -> np.ndarray:
    poisson = stats.poisson(mu=lambda_params)
    k_list = np.arange(k)
    return poisson.pmf(k=k_list)
