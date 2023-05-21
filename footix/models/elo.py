import numpy as np
import pandas as pd

from footix.models.abstract_model import CustomModel
from footix.models.teamElo import team
from footix.utils.decorators import verify_required_column
from footix.utils.utils import DICO_COMPATIBILITY


class EloDavidson(CustomModel):
    def __init__(
        self,
        n_teams: int,
        k0: int,
        lambd: float,
        sigma: int,
        agnostic_probs: list,
        **kwargs,
    ):
        super().__init__(n_teams, **kwargs)
        self.checkProbas(agnostic_probs)
        PH, PD, PA = agnostic_probs
        self.kappa = self.computeKappa(P_H=PH, P_D=PD, P_A=PA)
        self.eta = self.computeEta(P_H=PH, P_A=PA)
        self.k0 = k0
        self.lambda_ = lambd
        self.sigma = sigma
        self.championnat: dict[str, team] = {}

    @verify_required_column(column_names={"HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"})
    def fit(self, X_train: pd.DataFrame):
        clubs = np.sort(np.unique(np.concatenate([X_train["HomeTeam"], X_train["AwayTeam"]])))
        if len(clubs) != self.n_teams:
            raise ValueError(
                "Number of teams in the training dataset is not the same as in this class"
                "instanciation"
            )

        for club in clubs:
            self.championnat[club] = team(club)

        for idx, row in X_train.iterrows():
            Home = row["HomeTeam"]
            Away = row["AwayTeam"]
            result = self.correspondance_result(row["FTR"])
            gamma = np.abs(row["FTHG"] - row["FTAG"])
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

    def predict(
        self, HomeTeam: str, AwayTeam: str, cote_fdj: bool = True
    ) -> tuple[float, float, float]:
        if cote_fdj:
            Home = DICO_COMPATIBILITY[HomeTeam]
            Away = DICO_COMPATIBILITY[AwayTeam]
        else:
            Home = HomeTeam
            Away = AwayTeam
        return self.compute_proba(self.championnat[Home], self.championnat[Away])

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
