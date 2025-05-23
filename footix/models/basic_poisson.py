import logging

import numpy as np
import pandas as pd
import scipy.optimize as optimize

import footix.models.score_matrix as score_matrix
import footix.models.utils as model_utils
from footix.utils.decorators import verify_required_column

logger = logging.getLogger(name=__name__)


class BasicPoisson:
    def __init__(self, n_teams: int, n_goals: int) -> None:
        self.n_teams = n_teams
        self.n_goals = n_goals

    @verify_required_column(column_names={"home_team", "away_team", "ftr", "fthg", "ftag"})
    def fit(self, X_train: pd.DataFrame) -> None:
        self.dict_teams = self.mapping_team_index(X_train["home_team"])
        self._sanity_check(X_train["away_team"])
        goals_home, basis_home = model_utils.compute_goals_home_vectors(
            X_train, map_teams=self.dict_teams, nbr_team=self.n_teams
        )
        goals_away, basis_away = model_utils.compute_goals_away_vectors(
            X_train, map_teams=self.dict_teams, nbr_team=self.n_teams
        )
        optimization_result = optimize.minimize(
            basic_poisson_likelihood,
            x0=np.zeros(2 * self.n_teams + 1),
            args=(goals_home, goals_away, basis_home, basis_away, self.n_teams),
            constraints=[
                {"type": "eq", "fun": lambda x: np.sum(x[1 : self.n_teams + 1]) - self.n_teams},
                {"type": "eq", "fun": lambda x: np.sum(x[self.n_teams + 1 :]) + self.n_teams},
            ],
        )
        if not optimization_result.success:
            logger.warning("Minimization routine was not successful.")
        model_params = optimization_result.x
        self.gamma = model_params[0]
        self.alphas = tuple(model_params[1 : self.n_teams + 1])
        self.betas = tuple(model_params[self.n_teams + 1 :])

    def print_parameters(self) -> None:
        str_resume = f"Gamma = {self.gamma} \n"
        str_alpha = "".join(
            [f"alpha team-{team} = {self.alphas[idx]} \n" for team, idx in self.dict_teams.items()]
        )
        str_beta = "".join(
            [f"beta team-{team} = {self.betas[idx]} \n" for team, idx in self.dict_teams.items()]
        )
        print(str_resume + str_alpha + str_beta)

    def predict(self, home_team: str, away_team: str) -> score_matrix.GoalMatrix:
        if home_team not in self.dict_teams.keys():
            raise ValueError(f"Home team {home_team} is not in the list.")
        if away_team not in self.dict_teams.keys():
            raise ValueError(f"Away team {away_team} is not in the list.")
        i = self.dict_teams[home_team]
        j = self.dict_teams[away_team]
        lamb = np.exp(self.alphas[i] + self.betas[j] + self.gamma)
        mu = np.exp(self.alphas[j] + self.betas[i])
        return score_matrix.GoalMatrix(
            home_probs=model_utils.poisson_proba(lambda_param=lamb, k=self.n_goals),
            away_probs=model_utils.poisson_proba(lambda_param=mu, k=self.n_goals),
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


def basic_poisson_likelihood(
    params: np.ndarray,
    goals_home: np.ndarray,
    goals_away: np.ndarray,
    basis_home: np.ndarray,
    basis_away: np.ndarray,
    n_teams: int,
) -> float:
    gamma = params[0]
    alphas = params[1 : n_teams + 1]
    betas = params[n_teams + 1 :]
    log_lamdas = np.dot(basis_home, alphas) + np.dot(basis_away, betas) + gamma
    log_mus = np.dot(basis_away, alphas) + np.dot(basis_home, betas)
    lambdas = np.exp(log_lamdas)
    mus = np.exp(log_mus)
    log = lambdas + mus - goals_home * log_lamdas - goals_away * log_mus
    return np.sum(log)
