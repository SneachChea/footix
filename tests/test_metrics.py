import math

import numpy as np
import pytest

import footix.metrics as metrics


def test_entropy_uniform():
    res = metrics.incertity(probas=[1.0 / 3, 1.0 / 3, 1.0 / 3], outcome_idx=0)
    assert np.isclose(res, 1.0)


def test_rps():
    res_1 = metrics.rps(probas=[0.5, 0.2, 0.3], outcome_idx=0)
    res_2 = metrics.rps(probas=[0.5, 0.3, 0.2], outcome_idx=0)
    assert res_1 == 0.17
    assert res_2 == 0.145


def test_zscore():
    zscore, mu, sigma = metrics.zscore(
        probas=[0.5, 0.2, 0.3], rps_observed=0.17, seed=42, n_iter=1_000_000
    )
    assert math.isclose(zscore, -0.6547, rel_tol=1e-3)
    assert math.isclose(mu, 0.23, rel_tol=1e-3)
    assert math.isclose(sigma, 0.09165, rel_tol=1e-3)


def test_entropy_certain():
    res = metrics.incertity(probas=[1.0, 0.0, 0.0], outcome_idx=0)
    assert np.isclose(res, 0.0)


def test_common_classification_metrics():
    probas = [0.7, 0.2, 0.1]
    assert np.isclose(metrics.log_loss(probas, 0), -np.log(0.7))
    expected = np.mean(np.square(np.array(probas) - [1, 0, 0]))
    assert np.isclose(metrics.brier_score(probas, 0), expected)
    assert metrics.accuracy(probas, 0) == 1.0


@pytest.mark.parametrize("fn", [metrics.log_loss, metrics.brier_score, metrics.accuracy])
@pytest.mark.parametrize(
    "probas",
    [[], [0.5], [0.5, 0.5, -0.1], [0.5, np.nan], [float("inf"), 0.5], [0.0, 0.0]],
)
def test_classification_metrics_reject_invalid_probabilities(fn, probas):
    with pytest.raises(ValueError):
        fn(np.asarray(probas), 0)


@pytest.mark.parametrize("fn", [metrics.log_loss, metrics.brier_score, metrics.accuracy])
@pytest.mark.parametrize("outcome_idx", [-1, 5])
def test_classification_metrics_reject_out_of_range_outcome(fn, outcome_idx):
    with pytest.raises(ValueError):
        fn([0.5, 0.2, 0.3], outcome_idx)
