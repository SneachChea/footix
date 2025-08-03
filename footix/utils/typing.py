from typing import Any, NamedTuple, Protocol

import numpy as np

ArrayLikeF = list[float] | np.ndarray


class ProtoModel(Protocol):
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def predict(self, HomeTeam: str, AwayTeam: str) -> Any:
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


class SampleProbaResult(NamedTuple):
    """A NamedTuple representing the probability results for a match outcome.

    Attributes:
        proba_home (np.ndarray): Array of probabilities for the home team winning.
        proba_draw (np.ndarray): Array of probabilities for a draw.
        proba_away (np.ndarray): Array of probabilities for the away team winning.

    """

    proba_home: np.ndarray
    proba_draw: np.ndarray
    proba_away: np.ndarray
