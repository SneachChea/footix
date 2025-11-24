import logging

import numpy as np
import pandas as pd
import scipy.optimize as optimize

import footix.models.score_matrix as score_matrix
import footix.models.utils as model_utils
from footix.utils.decorators import verify_required_column

logger = logging.getLogger(name=__name__)

__all__ = ["PoissonModel"]


class PoissonModel:
    def __init__(self, n_teams: int, n_goals: int) -> None:
        if n_teams <= 0:
            raise ValueError("Number of teams should be positive")
        if n_goals <= 5:
            raise ValueError("Number of goals must be positive")
        self.n_teams = n_teams
        self.n_goals = n_goals

    @verify_required_column(
        column_names={"home_team", "away_team", "ftr", "fthg", "ftag"}
    )
    def fit(self, X_train: pd.DataFrame) -> None:
        """
        Fit the Poisson model to the training data.

        This method trains the model by estimating the parameters (gamma, alphas, betas)
        that maximize the likelihood of the observed goals in the training data.
        It performs optimization using scipy's minimize function with constraints
        on the sum of alpha and beta parameters.

        Args:
            X_train: DataFrame containing match data with columns:
                - home_team: Name of the home team
                - away_team: Name of the away team
                - ftr: Full-time result (unused in this implementation)
                - fthg: Full-time home goals
                - ftag: Full-time away goals

        Returns:
            None. The method updates the instance attributes:
            - gamma: Intercept parameter
            - alphas: Home team strength parameters
            - betas: Away team strength parameters
            - dict_teams: Mapping of team names to indices
        """
        self.dict_teams = self.mapping_team_index(X_train["home_team"])
        self._sanity_check(X_train["away_team"])
        goals_home, basis_home = model_utils.compute_goals_home_vectors(
            X_train, map_teams=self.dict_teams, nbr_team=self.n_teams
        )
        goals_away, basis_away = model_utils.compute_goals_away_vectors(
            X_train, map_teams=self.dict_teams, nbr_team=self.n_teams
        )
        optimization_result = optimize.minimize(
            self.basic_poisson_likelihood,
            x0=np.zeros(2 * self.n_teams + 1),
            args=(goals_home, goals_away, basis_home, basis_away),
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda x: np.sum(x[1 : self.n_teams + 1]) - self.n_teams,
                },
                {
                    "type": "eq",
                    "fun": lambda x: np.sum(x[self.n_teams + 1 :]) + self.n_teams,
                },
            ],
        )
        if not optimization_result.success:
            logger.warning("Minimization routine was not successful.")
        model_params = optimization_result.x
        self.gamma = model_params[0]
        self.alphas = tuple(model_params[1 : self.n_teams + 1])
        self.betas = tuple(model_params[self.n_teams + 1 :])

    def print_parameters(self) -> None:
        if not hasattr(self, "gamma"):
            raise AttributeError("Model not trained yet")
        str_resume = f"Gamma = {self.gamma} \n"
        str_alpha = "".join(
            [
                f"alpha team-{team} = {self.alphas[idx]} \n"
                for team, idx in self.dict_teams.items()
            ]
        )
        str_beta = "".join(
            [
                f"beta team-{team} = {self.betas[idx]} \n"
                for team, idx in self.dict_teams.items()
            ]
        )
        print(str_resume + str_alpha + str_beta)

    def predict(self, home_team: str, away_team: str) -> score_matrix.GoalMatrix:
        """Predict the goal probability matrix for a match between two teams.

        Uses the trained model parameters to calculate the expected number of goals
        for both teams, then computes the probability distribution of goals for each team
        using the Poisson distribution.

        Args:
            home_team: Name of the home team.
            away_team: Name of the away team.

        Returns:
            GoalMatrix containing the probability distributions for home and away team goals.

        Raises:
            ValueError: If either team is not in the list of trained teams.
        """
        if home_team not in self.dict_teams.keys():
            raise ValueError(f"Home team {home_team} is not in the list.")
        if away_team not in self.dict_teams.keys():
            raise ValueError(f"Away team {away_team} is not in the list.")
        i = self.dict_teams[home_team]
        j = self.dict_teams[away_team]
        lamb, mu = self.goal_expectation(home_team_id=i, away_team_id=j)
        return score_matrix.GoalMatrix(
            home_goals_probs=model_utils.poisson_proba(
                lambda_param=lamb, k=self.n_goals
            ),
            away_goals_probs=model_utils.poisson_proba(lambda_param=mu, k=self.n_goals),
        )

    def goal_expectation(
        self, home_team_id: int, away_team_id: int
    ) -> tuple[float, float]:
        """Calculate expected goals for home and away teams.

        The expected number of goals for the home team (lamb) and away team (mu)
        are calculated using the model parameters. The home team's expectation
        includes the intercept term gamma, while the away team's expectation does not.

        Args:
            home_team_id: Index of the home team in the model's parameter arrays.
            away_team_id: Index of the away team in the model's parameter arrays.

        Returns:
            Tuple containing expected goals for home team (lamb) and away team (mu).
        """
        lamb = np.exp(self.alphas[home_team_id] + self.betas[away_team_id] + self.gamma)
        mu = np.exp(self.alphas[away_team_id] + self.betas[home_team_id])
        return lamb, mu

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
            raise ValueError(
                f"Expecting {self.n_teams} teams, only got {len(self.dict_teams)}."
            )

    def basic_poisson_likelihood(
        self,
        params: np.ndarray,
        goals_home: np.ndarray,
        goals_away: np.ndarray,
        basis_home: np.ndarray,
        basis_away: np.ndarray,
    ) -> float:
        gamma = params[0]
        alphas = params[1 : self.n_teams + 1]
        betas = params[self.n_teams + 1 :]
        log_lamdas = np.dot(basis_home, alphas) + np.dot(basis_away, betas) + gamma
        log_mus = np.dot(basis_away, alphas) + np.dot(basis_home, betas)
        lambdas = np.exp(log_lamdas)
        mus = np.exp(log_mus)
        log = lambdas + mus - goals_home * log_lamdas - goals_away * log_mus
        return np.sum(log)
