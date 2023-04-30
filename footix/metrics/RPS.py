from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from footix.metrics.abc_metric import Metric
from footix.metrics.functional.metrics_function import rps


class RPS(Metric):
    _rps: list
    higher_is_better = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("_rps")

    def __call__(self, probas: Union[list, np.ndarray], outcome_idx: int) -> None:
        self._rps.append(rps(probas, outcome_idx))

    def compute(self) -> tuple[float, float]:
        return np.mean(self._rps), np.std(self._rps)  # type: ignore

    def visualize(self, n_bins: int):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(self._rps, bins=n_bins, edgecolor="black")
        ax.set_xlabel("RPS")
        ax.set_ylabel("Count")
        plt.show()
