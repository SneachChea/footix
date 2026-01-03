from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from footix.utils.typing import ArrayLikeF, ProbaResult


@dataclass
class GoalMatrix:
    """Utilities for match score probability matrices.

    `GoalMatrix` builds a joint distribution over football scores
    (home_goals, away_goals) from two marginal probability vectors.
    Optionally, a non-negative correlation/weight matrix can be applied element-wise
    to reweight scorelines.

    Notes:
        The input probability vectors are validated and normalized to sum to 1.

    Args:
        home_goals_probs: 1D array-like of non-negative goal probabilities for the home team.
        away_goals_probs: 1D array-like of non-negative goal probabilities for the away team.
        correlation_matrix: Optional 2D non-negative array of shape (n, n) applied
            element-wise to the outer product.

    Raises:
        ValueError: If inputs are not 1D, contain NaN/Inf, contain negative values,
            have incompatible lengths, have zero total probability mass, or if the
            correlation matrix is invalid.

    """

    home_goals_probs: ArrayLikeF
    away_goals_probs: ArrayLikeF
    correlation_matrix: np.ndarray | None = None
    matrix_array: np.ndarray = field(init=False)

    def __post_init__(self):
        self._checks_init()
        self.matrix_array = np.outer(self.home_goals_probs, self.away_goals_probs)
        if self.correlation_matrix is not None:
            self.matrix_array = self.matrix_array * self.correlation_matrix
            mass = float(np.sum(self.matrix_array))
            if (not np.isfinite(mass)) or (mass <= 0.0):
                raise ValueError(
                    "correlation_matrix produces zero or non-finite total probability mass"
                )
            self.matrix_array = self.matrix_array / mass

    def _checks_init(self) -> None:
        """Validate and normalize initialization inputs.

        Raises:
            ValueError: If any of the invariants described in the class docstring are violated.

        """

        self.home_goals_probs = np.asarray(self.home_goals_probs, dtype=float)
        self.away_goals_probs = np.asarray(self.away_goals_probs, dtype=float)

        if (self.home_goals_probs.ndim != 1) or (self.away_goals_probs.ndim != 1):
            raise ValueError("home_goals_probs and away_goals_probs must be 1D arrays")

        if self.home_goals_probs.size == 0:
            raise ValueError("home_goals_probs and away_goals_probs must be non-empty")

        if self.home_goals_probs.shape[0] != self.away_goals_probs.shape[0]:
            raise ValueError("home_goals_probs and away_goals_probs must have the same length")

        for name, arr in (
            ("home_goals_probs", self.home_goals_probs),
            ("away_goals_probs", self.away_goals_probs),
        ):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must contain only finite values")
            if np.any(arr < 0.0):
                raise ValueError(f"{name} must be non-negative")
            mass = float(np.sum(arr))
            if (not np.isfinite(mass)) or (mass <= 0.0):
                raise ValueError(f"{name} must have positive total probability mass")

        self.home_goals_probs = self.home_goals_probs / float(np.sum(self.home_goals_probs))
        self.away_goals_probs = self.away_goals_probs / float(np.sum(self.away_goals_probs))

        if self.correlation_matrix is not None:
            corr = np.asarray(self.correlation_matrix, dtype=float)
            if corr.ndim != 2:
                raise ValueError("correlation_matrix must be a 2D array")
            n = int(self.home_goals_probs.shape[0])
            if corr.shape != (n, n):
                raise ValueError(
                    "correlation_matrix must have shape (n, n) matching probabilities length"
                )
            if not np.all(np.isfinite(corr)):
                raise ValueError("correlation_matrix must contain only finite values")
            if np.any(corr < 0.0):
                raise ValueError("correlation_matrix must be non-negative")
            self.correlation_matrix = corr

    def return_probas(self) -> ProbaResult:
        """Return results probabilities in this order: home_win, draw, away_win.

        Returns:
            ProbaResult: NamedTuple of probabilities

        Raises:
            ValueError: If the internal probability matrix has zero mass.
        """
        home_win = np.sum(np.tril(self.matrix_array, -1))
        draw = np.sum(np.diag(self.matrix_array))
        away_win = np.sum(np.triu(self.matrix_array, 1))

        total = float(home_win + draw + away_win)
        if (not np.isfinite(total)) or (total <= 0.0):
            raise ValueError("matrix_array must have positive total probability mass")
        if not np.isclose(total, 1.0, rtol=1e-12, atol=1e-12):
            home_win, draw, away_win = home_win / total, draw / total, away_win / total
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

    def assert_format_15(self) -> None:
        if len(self.home_goals_probs) < 2:
            raise ValueError(
                "home_goals_probs and away_goals_probs must have length >= 2 for less_15_goals"
            )

    def assert_format_25(self) -> None:
        if len(self.home_goals_probs) < 3:
            raise ValueError(
                "home_goals_probs and away_goals_probs must have length >= 3 for less_25_goals"
            )

    def visualize(self, n_goals: int = 5) -> None:
        if n_goals > len(self.home_goals_probs):
            raise ValueError(
                f"n_goals must be <= len(home_goals_probs) ({len(self.home_goals_probs)})\n"
                f"got {n_goals}"
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
