from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from footix.utils.typing import ArrayLikeF, ProbaResult


@dataclass
class GoalMatrix:
    """Dataclass that compile all functions related to probability from "results" models
    (Bayesian, Dixon, Poisson, etc.)"""

    home_goals_probs: ArrayLikeF
    away_goals_probs: ArrayLikeF
    correlation_matrix: np.ndarray | None = None
    matrix_array: np.ndarray = field(init=False)

    def __post_init__(self):
        self._checks_init()
        self.matrix_array = np.outer(self.home_goals_probs, self.away_goals_probs)
        if self.correlation_matrix is not None:
            self.matrix_array = self.matrix_array * self.correlation_matrix
            self.matrix_array = self.matrix_array / np.sum(self.matrix_array)

    def _checks_init(self):
        self.home_goals_probs = np.asarray(self.home_goals_probs)
        self.away_goals_probs = np.asarray(self.away_goals_probs)
        if (self.home_goals_probs.ndim > 1) or (self.away_goals_probs.ndim > 1):
            raise ValueError("Array probs should be one dimensional")
        if len(self.home_goals_probs) != len(self.away_goals_probs):
            raise ValueError("Length of proba's array should be the same")
        if self.correlation_matrix is not None:
            if self.home_goals_probs.shape[0] != self.correlation_matrix.shape[0]:
                raise ValueError(
                    "Size between probability matrix and correlation matrix should be the same"
                )

    def return_probas(self) -> ProbaResult:
        """Return results probabilities in this order: home_win, draw, away_win.

        Returns:
            ProbaResult: NamedTuple of probabilities
        """
        home_win = np.sum(np.tril(self.matrix_array, -1))
        draw = np.sum(np.diag(self.matrix_array))
        away_win = np.sum(np.triu(self.matrix_array, 1))

        tail_mass = 1.0 - self.matrix_array.sum()

        if tail_mass > 0:
            s = home_win + draw + away_win
            home_win, draw, away_win = (
                home_win + tail_mass * (home_win / s),
                draw + tail_mass * (draw / s),
                away_win + tail_mass * (away_win / s),
            )
        return ProbaResult(proba_home=home_win, proba_draw=draw, proba_away=away_win)

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
        if len(self.home_goals_probs) < 2:
            raise TypeError("Probas should be longer than 3")

    def assert_format_25(self):
        if len(self.home_goals_probs) < 3:
            raise TypeError("Probas should be longer than 4")

    def visualize(self, n_goals: int = 5) -> None:
        if n_goals > len(self.home_goals_probs):
            raise ValueError(
                f"Requested n_goals={n_goals} exceeds available goal probabilities "
                f"({len(self.home_goals_probs)})."
            )
        tmp_small = self.matrix_array[:n_goals, :n_goals]
        _, ax = plt.subplots()
        ax.matshow(tmp_small, cmap="coolwarm")
        for i in range(len(tmp_small)):
            for j in range(len(tmp_small)):
                ax.text(j, i, round(tmp_small[i, j], 3), ha="center", va="center", color="w")
        ax.set_xlabel("Away team")
        ax.set_ylabel("Home team")
        plt.show()

    def asian_handicap_results(self, handicap: float) -> ProbaResult:
        """Calculate the probabilities for a home win, draw, and away win after applying an Asian
        handicap using vectorized operations. The handicap is added to the home team's goal count.

        Args:
            handicap (float): The handicap to be applied to the home team's score.
        Returns:
            ProbaResult: home_win, draw, away_win probabilities.

        """
        n = len(self.home_goals_probs)
        tol = 1e-6  # tolerance for float equality

        # Create a grid of differences between home and away goals
        home_indices = np.arange(n)[:, None] + handicap  # Add handicap to home goals
        away_indices = np.arange(n)
        diff_matrix = home_indices - away_indices

        # Calculate probabilities based on the difference matrix
        home_win = np.sum(self.matrix_array[diff_matrix > tol])
        away_win = np.sum(self.matrix_array[diff_matrix < -tol])
        draw = np.sum(self.matrix_array[np.abs(diff_matrix) <= tol])

        return ProbaResult(proba_home=home_win, proba_draw=draw, proba_away=away_win)

    def __str__(self) -> str:
        home_str = ", ".join(f"{x:.2f}" for x in self.home_goals_probs[:5])
        away_str = ", ".join(f"{x:.2f}" for x in self.away_goals_probs[:5])
        return f"Goal Matrix computed using [{home_str}, ...] and [{away_str}, ...]."

    def get_probable_score(self) -> tuple[int, int]:
        """Return the most probable score (home_goals, away_goals) based on the matrix_array.

        Returns
        -------
        tuple of int
            The (home_goals, away_goals) corresponding to the highest probability in matrix_array.

        Examples
        --------
        >>> gm = GoalMatrix(home_goals_probs, away_goals_probs)
        >>> gm.get_probable_score()
        (2, 1)

        """
        idx = np.unravel_index(np.argmax(self.matrix_array), self.matrix_array.shape)
        return int(idx[0]), int(idx[1])

    def double_chance(self) -> tuple[float, float, float]:
        """Calculates the double chance probabilities for a football match outcome.

        Double chance is a betting market that covers two of the three possible outcomes
        in a match:
            - Home win or Draw (1X)
            - Draw or Away win (X2)
            - Home win or Away win (12)

        Returns:
            tuple[float, float, float]: A tuple containing:
                - Probability of Home win or Draw (1X)
                - Probability of Draw or Away win (X2)
                - Probability of Home win or Away win (12)

        """
        probas = self.return_probas()
        p_1_x = probas.proba_home + probas.proba_draw
        p_x_2 = probas.proba_draw + probas.proba_away
        p_1_2 = probas.proba_home + probas.proba_away
        return p_1_x, p_x_2, p_1_2

    def probability_both_teams_scores(self) -> float:
        return np.sum(self.matrix_array[1:, 1:])
