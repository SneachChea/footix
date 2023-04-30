from dataclasses import dataclass, field
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class GoalMatrix:
    home_probs: np.ndarray
    away_probs: np.ndarray
    m: np.ndarray = field(init=False)

    def __post_init__(self):
        self._assert_init()
        self.m = np.outer(self.home_probs, self.away_probs)

    def _assert_init(self):
        if (self.home_probs.ndim > 1) or (self.away_probs.ndim > 1):
            raise TypeError("Array probs should be one dimensional")
        if len(self.home_probs) != len(self.away_probs):
            raise TypeError("Length of proba's array should be the same")

    def return_probas(self) -> Tuple[float, float, float]:
        hom_win = np.sum(np.tril(self.m, -1))
        draw = np.sum(np.diag(self.m))
        away_win = np.sum(np.triu(self.m, 1))
        return hom_win, draw, away_win

    def less_15_goals(self) -> float:
        self.assert_format_15()
        return self.m[0, 0] + self.m[0, 1] + self.m[1, 0]

    def less_25_goals(self) -> float:
        self.assert_format_25()
        return self.less_15_goals() + self.m[0, 2] + self.m[1, 1] + self.m[2, 0]

    def more_25_goals(self) -> float:
        return 1 - self.less_25_goals()

    def more_15_goals(self) -> float:
        return 1.0 - self.less_15_goals()

    def assert_format_15(self):
        if len(self.home_probs) < 2:
            raise TypeError("Probas should be larger than 3")

    def assert_format_25(self):
        if len(self.home_probs) < 3:
            raise TypeError("Probas should be larger than 4")

    def visualize(self) -> None:
        fig, ax = plt.subplots()
        im = ax.matshow(self.m, cmap="coolwarm")
        for i in range(len(self.home_probs)):
            for j in range(len(self.away_probs)):
                text = ax.text(j, i, round(self.m[i, j], 3),
                            ha="center", va="center", color="w")
        ax.set_xlabel("Away team")
        ax.set_ylabel("Home team")