import logging

import numpy as np
import pandas as pd
import scipy.optimize as optimize
import scipy.stats as stats

import footix.models.score_matrix as score_matrix
import footix.models.utils as model_utils

logger = logging.getLogger(name=__name__)


class DixonColes:
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
            dixon_coles_likelihood,
            x0=0.1 * np.ones(2 * self.n_teams + 2),
            args=(goals_home, goals_away, basis_home, basis_away, self.n_teams),
            constraints=[
                {"type": "eq", "fun": lambda x: np.sum(x[2 : self.n_teams + 2]) - self.n_teams},
                {"type": "eq", "fun": lambda x: np.sum(x[self.n_teams + 2 :]) + self.n_teams},
            ],
        )
        if not optimization_result.success:
            logger.warning("Minimization routine was not successful.")
        model_params = optimization_result.x
        self.gamma = model_params[0]
        self.rho = model_params[1]
        self.alphas = tuple(model_params[2 : self.n_teams + 2])
        self.betas = tuple(model_params[self.n_teams + 2 :])

    def print_parameters(self) -> None:
        str_gamma = f"Gamma = {self.gamma}\n"
        str_rho = f"rho = {self.rho}\n"
        str_alpha = "".join(
            [f"alpha team-{team} = {self.alphas[idx]}\n" for team, idx in self.dict_teams.items()]
        )
        str_beta = "".join(
            [f"beta team-{team} = {self.betas[idx]}\n" for team, idx in self.dict_teams.items()]
        )
        print(str_gamma + str_rho + str_alpha + str_beta)

    def predict(self, home_team: str, away_team: str) -> score_matrix.GoalMatrix:
        if home_team not in self.dict_teams.keys():
            raise ValueError(f"Home team {home_team} is not in the list.")
        if away_team not in self.dict_teams.keys():
            raise ValueError(f"Away team {away_team} is not in the list.")
        i = self.dict_teams[home_team]
        j = self.dict_teams[away_team]
        lamb = np.exp(self.alphas[i] + self.betas[j] + self.gamma)
        mu = np.exp(self.alphas[j] + self.betas[i])
        rho_correction = matrix_rho(self.rho, lam=lamb, mu=mu, size=self.n_goals)
        return score_matrix.GoalMatrix(
            home_probs=poisson_proba(lambda_params=lamb, k=self.n_goals),
            away_probs=poisson_proba(lambda_params=mu, k=self.n_goals),
            correlation_matrix=rho_correction,
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


def dixon_coles_likelihood(
    params: np.ndarray,
    goals_home: np.ndarray,
    goals_away: np.ndarray,
    basis_home: np.ndarray,
    basis_away: np.ndarray,
    n_teams: int,
) -> float:
    gamma = params[0]
    rho = params[1]
    alphas = params[2 : n_teams + 2]
    betas = params[n_teams + 2 :]
    log_lamdas = np.dot(basis_home, alphas) + np.dot(basis_away, betas) + gamma
    log_mus = np.dot(basis_away, alphas) + np.dot(basis_home, betas)
    lambdas = np.exp(log_lamdas)
    mus = np.exp(log_mus)
    log = (
        lambdas
        + mus
        - goals_home * log_lamdas
        - goals_away * log_mus
        - np.log(rho_correction_vec(rho, x=goals_home, y=goals_away, lam=lambdas, mu=mus) + 1e-8)
    )
    return np.sum(log)


def rho_correction_vec(
    rho: np.ndarray, x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray
) -> np.ndarray:
    dc_adj = np.select(
        [
            (x == 0) & (y == 0),
            (x == 0) & (y == 1),
            (x == 1) & (y == 0),
            (x == 1) & (y == 1),
        ],
        [
            1 - (lam * mu * rho),
            1 + (lam * rho),
            1 + (mu * rho),
            1 - rho,
        ],
        default=1.,
    )
    return dc_adj


def poisson_proba(lambda_params: float, k: int) -> np.ndarray:
    poisson = stats.poisson(mu=lambda_params)
    k_list = np.arange(k)
    return poisson.pmf(k=k_list)


def matrix_rho(rho: float, lam: float, mu: float, size: int) -> np.ndarray:
    one_matrix = np.ones((size, size))
    one_matrix[0, 0] = 1 - lam * mu * rho
    one_matrix[0, 1] = 1 + (lam * rho)
    one_matrix[1, 0] = 1 + (mu * rho)
    one_matrix[1, 1] = 1 - rho
    return one_matrix
