import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_goal_matrix(matrix: np.ndarray, n_goals: int = 5) -> Figure:
    """Visualize the probability matrix of match scores.

    Args:
        matrix: 2D array of score probabilities.
        n_goals: Number of goals to show on each axis.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    if n_goals > matrix.shape[0]:
        n_goals = matrix.shape[0]

    tmp_small = matrix[:n_goals, :n_goals]
    fig, ax = plt.subplots()
    ax.matshow(tmp_small, cmap="coolwarm")
    for i in range(len(tmp_small)):
        for j in range(len(tmp_small)):
            ax.text(j, i, round(tmp_small[i, j], 3), ha="center", va="center", color="w")
    ax.set_xlabel("Away team")
    ax.set_ylabel("Home team")
    return fig
