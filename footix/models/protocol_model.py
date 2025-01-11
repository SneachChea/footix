from typing import Any, Protocol
import pandas as pd
import footix.models.score_matrix as score_matrix

class ProtoModel(Protocol):
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def predict(self, HomeTeam: str, AwayTeam: str) -> Any:
        ...


class ProtoPoisson(Protocol):
    def __init__(self, n_teams: int, n_goals: int)-> None:
        ...

    def fit(self, X_train: pd.DataFrame)-> None:
        ...

    def predict(self, home_team: str, away_team: str)-> score_matrix.GoalMatrix:
        ...