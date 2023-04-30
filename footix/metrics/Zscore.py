from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np

from footix.metrics.abc_metric import Metric
from footix.metrics.functional.metrics_function import rps, zscore


class Zscore(Metric):
    higher_is_better: bool = False
    _zscore: List

    def __init__(self) -> None:
        super().__init__()
        self.add_state("_zscore")

    def __call__(self, probas: Union[list, np.ndarray], outcome_idx: int) -> None:
        rps_real = rps(probas, outcome_idx)
        self._zscore.append(zscore(probas, rps_real))

    def compute(self) -> tuple[float, float]:
        return np.mean(self._zscore), np.std(self._zscore)  # ignore: type

    def visualize(self, n_bins: int):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(self._zscore, bins=n_bins, edgecolor="black")
        ax.set_xlabel("Z-score")
        ax.set_ylabel("Count")
        plt.show()
