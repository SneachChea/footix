import numpy as np
from typing import Union, Any
from .functional import entropy
from .abc_metric import Metric
import matplotlib.pyplot as plt


class Entropy(Metric):
    _entropy: list
    higher_is_better = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("_entropy")

    def __call__(self, probas: Union[list, np.ndarray], outcome_idx: int) -> None:
        self._entropy.append(entropy(probas, outcome_idx))

    def compute(self) -> tuple[float, float]:
        return np.mean(self._entropy), np.std(self._entropy)

    def visualize(self, n_bins: int):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(self._entropy, bins=n_bins, edgecolor="black")
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Count")
        plt.show()
