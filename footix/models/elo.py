import numpy as np
import pandas as pd

from footix.data_io.data_reader import EloDataReader
from footix.models.team_elo import team


# TODO: A dataclass for agnostic_probs?
class EloDavidson:
    def __init__(
        self,
        n_teams: int,
        k0: int,
        lambd: float,
        sigma: int,
        agnostic_probs: list,
        **kwargs,
    ):
        self.n_teams = n_teams
        self.checkProbas(agnostic_probs)
        PH, PD, PA = agnostic_probs
        self.kappa = self.computeKappa(P_H=PH, P_D=PD, P_A=PA)
        self.eta = self.computeEta(P_H=PH, P_A=PA)
        self.k0 = k0
        self.lambda_ = lambd
        self.sigma = sigma
        self.championnat: dict[str, team] = {}

    def fit(self, X_train: pd.DataFrame | EloDataReader):
        if isinstance(X_train, pd.DataFrame):
            X_train = EloDataReader(df_data=X_train)
        clubs = X_train.unique_teams()
        if len(clubs) != self.n_teams:
            raise ValueError(
                "Number of teams in the training dataset is not the same as in this class"
                "instanciation"
            )

        for club in clubs:
            self.championnat[club] = team(club)

        for game in X_train:
            Home = game.home_team
            Away = game.away_team
            result = self.correspondance_result(game.result)
            gamma = np.abs(game.home_goals - game.away_goals)
            K = self.K(gamma)
            self.update_rank(self.championnat[Home], self.championnat[Away], result, K)

    def reset(self):
        self.championnat = {}

    @staticmethod
    def computeKappa(P_H: float, P_D: float, P_A: float) -> float:
        return P_D / np.sqrt(P_H * P_A)

    @staticmethod
    def computeEta(P_H: float, P_A: float) -> float:
        return np.log10(P_H / P_A)

    @staticmethod
    def checkProbas(agnostic_probs: list) -> None:
        if not np.isclose(np.sum(agnostic_probs), b=1.0):
            raise ValueError("Probabilities do not sum to one.\n")

    def K(self, gamma: int) -> float:
        return self.k0 * (1.0 + gamma) ** self.lambda_

    @staticmethod
    def correspondance_result(result: str) -> float:
        if result not in ["D", "H", "A"]:
            raise ValueError("result must be 'D', 'H' or 'A'")
        if result == "D":
            return 0.5
        if result == "H":
            return 1.0
        return 0.0

    def estimated_res(self, difference: float) -> float:
        denom = 0.5 * difference / self.sigma
        return (10**denom + 0.5 * self.kappa) / (10**denom + 10 ** (-denom) + self.kappa)

    def update_rank(self, teamH: team, teamA: team, result: float, K: float) -> None:
        diff_rank = teamH.rank - teamA.rank + self.eta * self.sigma
        new_rankH = teamH.rank + K * (result - self.estimated_res(diff_rank))
        new_rankA = teamA.rank + K * (1.0 - result - self.estimated_res(-diff_rank))
        teamH.rank = new_rankH
        teamA.rank = new_rankA

    def __str__(self):
        if hasattr(self, "championnat"):
            classement = ""
            sorted_championnat = {
                k: v for k, v in sorted(self.championnat.items(), key=lambda item: -item[1].rank)
            }
            for i, k in enumerate(sorted_championnat.keys()):
                classement += f"{i+1}. {k} : {sorted_championnat[k].rank} \n"
            return classement
        else:
            return "{}"

    def predict(self, home_team: str, away_team: str) -> tuple[float, float, float]:
        return self.compute_proba(self.championnat[home_team], self.championnat[away_team])

    def probaW(self, diff: float) -> float:
        num = 0.5 * diff / self.sigma
        return 10 ** (num) / (10**num + 10 ** (-num) + self.kappa)

    def probaD(self, diff: float) -> float:
        num = 0.5 * diff / self.sigma
        return self.kappa / (10**num + 10 ** (-num) + self.kappa)

    def compute_proba(self, teamH, teamA) -> tuple[float, float, float]:
        diff = teamH.rank - teamA.rank
        diff = diff + self.eta * self.sigma
        probaH = self.probaW(diff)
        probaA = self.probaW(-diff)
        probaDraw = self.probaD(diff)
        return probaH, probaDraw, probaA
