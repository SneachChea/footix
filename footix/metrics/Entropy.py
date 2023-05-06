import matplotlib.pyplot as plt
import numpy as np

from footix.metrics.abc_metric import Metric
from footix.metrics.functional.metrics_function import entropy


class Entropy(Metric):
    _entropy: list[float]
    higher_is_better = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("_entropy")

    def __call__(self, probas: list | np.ndarray, outcome_idx: int) -> None:
        self._entropy.append(entropy(probas, outcome_idx))

    def compute(self) -> tuple[float, float]:
        return np.mean(self._entropy), np.std(self._entropy)  # ignore: type

    def visualize(self, n_bins: int):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(self._entropy, bins=n_bins, edgecolor="black")
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Count")
        plt.show()
