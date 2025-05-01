import math

import numpy as np

import footix.metrics as metrics


def test_entropy():
    res = metrics.incertity(probas=[1.0 / 3, 1.0 / 3, 1.0 / 3], outcome_idx=0)
    assert np.isclose(res, 1.0)


def test_rps():
    res_1 = metrics.rps(probas=[0.5, 0.2, 0.3], outcome_idx=0)
    res_2 = metrics.rps(probas=[0.5, 0.3, 0.2], outcome_idx=0)
    assert res_1 == 0.17
    assert res_2 == 0.145


def test_zscore():
    zscore, mu, sigma = metrics.zscore(
        probas=[0.5, 0.2, 0.3], rps_real=0.17, seed=42, n_iter=10_000
    )
    assert math.isclose(zscore, -0.6455544163998762)
    assert math.isclose(mu, 0.22884000000000002)
    assert math.isclose(sigma, 0.09113646032187116)
