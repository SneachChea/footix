from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np


# TODO: Add a dataclass for probabilities ?
@dataclass
class GoalMatrix:
    """Dataclass that compile all functions related to probability from Poisson models (Bayesian,
    Dixon, etc.)"""

    home_probs: np.ndarray
    away_probs: np.ndarray
    correlation_matrix: np.ndarray | None = None
    matrix_array: np.ndarray = field(init=False)

    def __post_init__(self):
        self._assert_init()
        self.matrix_array = np.outer(self.home_probs, self.away_probs)
        if self.correlation_matrix is not None:
            self.matrix_array = self.matrix_array * self.correlation_matrix
            self.matrix_array = self.matrix_array / np.sum(self.matrix_array)

    def _assert_init(self):
        if (self.home_probs.ndim > 1) or (self.away_probs.ndim > 1):
            raise TypeError("Array probs should be one dimensional")
        if len(self.home_probs) != len(self.away_probs):
            raise TypeError("Length of proba's array should be the same")
        if self.correlation_matrix is not None:
            if self.home_probs.shape[0] != self.correlation_matrix.shape[0]:
                raise ValueError(
                    "Size between probability matrix and correlation matrix should be the same"
                )

    def return_probas(self) -> tuple[float, float, float]:
        """Return results probabilities in this order: home_win, draw, away_win.

        Returns:
            tuple[float, float, float]: _description_
        """
        home_win = np.sum(np.tril(self.matrix_array, -1))
        draw = np.sum(np.diag(self.matrix_array))
        away_win = np.sum(np.triu(self.matrix_array, 1))
        return home_win, draw, away_win

    def less_15_goals(self) -> float:
        self.assert_format_15()
        return self.matrix_array[0, 0] + self.matrix_array[0, 1] + self.matrix_array[1, 0]

    def less_25_goals(self) -> float:
        self.assert_format_25()
        return (
            self.less_15_goals()
            + self.matrix_array[0, 2]
            + self.matrix_array[1, 1]
            + self.matrix_array[2, 0]
        )

    def more_25_goals(self) -> float:
        return 1 - self.less_25_goals()

    def more_15_goals(self) -> float:
        return 1.0 - self.less_15_goals()

    def assert_format_15(self):
        if len(self.home_probs) < 2:
            raise TypeError("Probas should be longer than 3")

    def assert_format_25(self):
        if len(self.home_probs) < 3:
            raise TypeError("Probas should be longer than 4")

    def visualize(self) -> None:
        _, ax = plt.subplots()
        ax.matshow(self.matrix_array, cmap="coolwarm")
        for i in range(len(self.home_probs)):
            for j in range(len(self.away_probs)):
                ax.text(
                    j, i, round(self.matrix_array[i, j], 3), ha="center", va="center", color="w"
                )
        ax.set_xlabel("Away team")
        ax.set_ylabel("Home team")
        plt.show()

    def asian_handicap_results(self, handicap: float) -> tuple[float, float, float]:
        """Calculate the probabilities for a home win, draw, and away win after applying an Asian
        handicap. The handicap is added to the home team's goal count.

        Args:
            handicap (float): The handicap to be applied to the home team's score.
        Returns:
            tuple[float, float, float]: home_win, draw, away_win probabilities.

        """
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        n = len(self.home_probs)
        tol = 1e-6  # tolerance for float equality
        for i in range(n):
            for j in range(n):
                diff = (i + handicap) - j
                if diff > tol:
                    home_win += self.matrix_array[i, j]
                elif diff < -tol:
                    away_win += self.matrix_array[i, j]
                else:
                    draw += self.matrix_array[i, j]
        return home_win, draw, away_win
