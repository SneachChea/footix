import footix.metrics.functional.metrics_function as fn_metrics
import numpy as np


def test_entropy_fn():
    res = fn_metrics.entropy(proba=[1./3, 1./3, 1./3], outcome_idx=0)
    assert np.isclose(res, 1.0)

def test_rps_fn():
    res_1 = fn_metrics.rps(probas=[0.5, 0.2, 0.3], outcome_idx=0)
    res_2 = fn_metrics.rps(probas=[0.5, 0.3, 0.2], outcome_idx=0)
    assert res_1 == 0.17
    assert res_2 == 0.145