from typing import Any, NamedTuple, Protocol

import numpy as np
import pandas as pd

import footix.models.score_matrix as score_matrix

ArrayLikeF = list[float] | np.ndarray


class ProtoModel(Protocol):
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def predict(self, HomeTeam: str, AwayTeam: str) -> Any:
        ...


class ProtoPoisson(Protocol):
    def __init__(self, n_teams: int, n_goals: int) -> None:
        ...

    def fit(self, X_train: pd.DataFrame) -> None:
        ...

    def predict(self, home_team: str, away_team: str) -> score_matrix.GoalMatrix:
        ...


class ProtoBayes(Protocol):
    def __init__(self, n_teams: int, n_goals: int) -> None:
        ...

    def fit(self, X_train: pd.DataFrame) -> None:
        ...

    def predict(self, home_team: str, away_team: str, **kwargs: Any) -> score_matrix.GoalMatrix:
        ...

    def get_samples(
        self, home_team: str, away_team: str, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


class RPSResult(NamedTuple):
    """Named tuple for Ranked Probability Score statistics."""

    z_score: float
    mean: float
    std_dev: float


class ProbaResult(NamedTuple):
    """Named tuple for Probabilities."""

    proba_home: float
    proba_draw: float
    proba_away: float
