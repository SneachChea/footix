import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize

from footix.models.abstract_model import CustomModel
from footix.utils.decorators import verify_required_column
from footix.utils.utils import DICO_COMPATIBILITY, EPS


class Poisson(CustomModel):
    def __init__(self, n_teams: int, **kwargs):
        super().__init__(n_teams, **kwargs)
        self.params = {}

    @verify_required_column(column_names={"HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"})
    def fit(self, X_train: pd.DataFrame, weighted: bool) -> None:
        teams = np.sort(np.unique(np.concatenate([X_train["HomeTeam"], X_train["AwayTeam"]])))
        if len(teams) != self.n_teams:
            raise ValueError(
                "Number of teams in the training dataset is not the same as in this class"
                "instanciation"
            )

        params = np.concatenate(
            (
                np.random.uniform(0.5, 1.5, (self.n_teams)),  # attack strength
                np.random.uniform(0, -1, (self.n_teams)),  # defence strength
                np.random.uniform(0, 1, (self.n_teams)),  # home advantage
                [-0.1],  # rho
            )
        )

        def _fit(params, df, teams):
            attack_params = dict(zip(teams, params[: self.n_teams]))
            defence_params = dict(zip(teams, params[self.n_teams : (2 * self.n_teams)]))
            home_advantage = dict(zip(teams, params[(2 * self.n_teams) : (3 * self.n_teams)]))
            rho = params[-1]

            llk = list()
            if weighted:
                for _, row in df.iterrows():
                    tmp = log_likelihood(
                        row["FTHG"],
                        row["FTAG"],
                        attack_params[row["HomeTeam"]],
                        defence_params[row["HomeTeam"]],
                        attack_params[row["AwayTeam"]],
                        defence_params[row["AwayTeam"]],
                        home_advantage[row["HomeTeam"]],
                        rho,
                        row["weight"],
                    )
                    llk.append(tmp)
            else:
                for _, row in df.iterrows():
                    tmp = log_likelihood(
                        row["FTHG"],
                        row["FTAG"],
                        attack_params[row["HomeTeam"]],
                        defence_params[row["HomeTeam"]],
                        attack_params[row["AwayTeam"]],
                        defence_params[row["AwayTeam"]],
                        home_advantage[row["HomeTeam"]],
                        rho,
                    )
                    llk.append(tmp)

            return np.sum(llk)

        options = {
            "maxiter": 200,
            "disp": False,
        }

        constraints = [{"type": "eq", "fun": lambda x: sum(x[: self.n_teams]) - self.n_teams}]

        res = minimize(
            _fit,
            params,
            args=(X_train, teams),
            constraints=constraints,
            options=options,
        )

        model_params = dict(
            zip(
                ["attack_" + team for team in teams]
                + ["defence_" + team for team in teams]
                + ["home_adv_" + team for team in teams]
                + ["rho"],
                res["x"],
            )
        )

        print("Log Likelihood: ", res["fun"])

        self.params = model_params

    def predict(
        self,
        HomeTeam: str,
        AwayTeam: str,
        score_matrix: bool = False,
        cote_fdj: bool = True,
    ) -> tuple[float, np.ndarray] | tuple:
        if not bool(self.params):
            raise AttributeError("Model is not trained. Please train it.")
        if cote_fdj:
            home_team = DICO_COMPATIBILITY[HomeTeam]
            away_team = DICO_COMPATIBILITY[AwayTeam]
        else:
            home_team = HomeTeam
            away_team = AwayTeam
        home_attack = self.params["attack_" + home_team]
        home_defence = self.params["defence_" + home_team]
        away_attack = self.params["attack_" + away_team]
        away_defence = self.params["defence_" + away_team]
        home_advantage = self.params["home_adv_" + home_team]
        rho = self.params["rho"]

        home_goal_expectation = np.exp(home_attack + away_defence + home_advantage)
        away_goal_expectation = np.exp(away_attack + home_defence)

        home_probs = stats.poisson.pmf(range(6), home_goal_expectation)
        away_probs = stats.poisson.pmf(range(6), away_goal_expectation)

        m = np.outer(home_probs, away_probs)

        m[0, 0] *= 1 - home_goal_expectation * away_goal_expectation * rho
        m[0, 1] *= 1 + home_goal_expectation * rho
        m[1, 0] *= 1 + away_goal_expectation * rho
        m[1, 1] *= 1 - rho

        home = np.sum(np.tril(m, -1))
        draw = np.sum(np.diag(m))
        away = np.sum(np.triu(m, 1))
        if score_matrix:
            return (home, draw, away), m
        return home, draw, away


def dc_decay(xi, t):
    return np.exp(-xi * t)


def rho_correction(
    goals_home: int, goals_away: int, home_exp: int, away_exp: int, rho: float
) -> float:
    if goals_home == 0 and goals_away == 0:
        return 1 - (home_exp * away_exp * rho)
    elif goals_home == 0 and goals_away == 1:
        return 1 + (home_exp * rho)
    elif goals_home == 1 and goals_away == 0:
        return 1 + (away_exp * rho)
    elif goals_home == 1 and goals_away == 1:
        return 1 - rho
    else:
        return 1.0


def log_likelihood(
    goals_home_observed: int,
    goals_away_observed: int,
    home_attack: float,
    home_defence: float,
    away_attack: float,
    away_defence: float,
    home_advantage: float,
    rho: float,
    weight: float | None = None,
) -> float:
    goal_expectation_home = np.exp(home_attack + away_defence + home_advantage)
    goal_expectation_away = np.exp(away_attack + home_defence)

    home_llk = stats.poisson.pmf(goals_home_observed, goal_expectation_home)
    away_llk = stats.poisson.pmf(goals_away_observed, goal_expectation_away)
    adj_llk = rho_correction(
        goals_home_observed,
        goals_away_observed,
        goal_expectation_home,
        goal_expectation_away,
        rho,
    )

    if goal_expectation_home < 0 or goal_expectation_away < 0 or adj_llk < 0:
        return 10000
    if weight is not None:
        log_llk = weight * (
            np.log(home_llk + EPS) + np.log(away_llk + EPS) + np.log(adj_llk + EPS)
        )
    else:
        log_llk = np.log(home_llk + EPS) + np.log(away_llk + EPS) + np.log(adj_llk + EPS)

    return -log_llk
