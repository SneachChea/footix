import numpy as np

import footix.implied as implied


def test_multiplicative_method_1D():
    odds = [5.4, 4.3, 1.55]
    proba, margin = implied.multiplicative(odds)
    assert np.allclose(proba, [0.1742, 0.2188, 0.6070], rtol=1e-3)
    assert np.isclose(margin, 0.0629, rtol=1e-3)


def test_multiplicative_method_2D():
    odds = np.array([[5.4, 4.3, 1.55]])
    proba, margin = implied.multiplicative(odds)
    assert np.allclose(proba, [[0.1742, 0.2188, 0.6070]], rtol=1e-3)
    assert np.allclose(margin, [0.0629], rtol=1e-3)


def test_power_method():
    odds = [5.4, 4.3, 1.55]
    proba, margin = implied.power(odds)
    assert np.allclose(proba, [0.1645, 0.2099, 0.6256], rtol=1e-3)
    assert np.isclose(margin, 0.0629, rtol=1e-3)


def test_shin_method():
    odds = [5.4, 4.3, 1.55]
    proba, margin = implied.shin(odds)
    assert np.allclose(proba, [0.1668, 0.2134, 0.6198], rtol=1e-3)
    assert np.isclose(margin, 0.0629, rtol=1e-3)
