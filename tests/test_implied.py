import numpy as np
import pytest

import footix.implied_odds as implied_odds


@pytest.fixture
def odds():
    return [5.4, 4.3, 1.55]

@pytest.fixture
def large_odds():
    return [1000, 2000, 3000]



class TestMultiplicativeMethod:
    def test_multiplicative_method_1D(self, odds):
        proba, margin = implied_odds.multiplicative_method(odds)
        assert np.allclose(proba, [0.1742, 0.2188, 0.6070], rtol=1e-3)
        assert np.isclose(margin, 0.0629, rtol=1e-3)

    def test_multiplicative_method_2D(self, odds):
        odds = np.array([odds])
        proba, margin = implied_odds.multiplicative_method(odds)
        assert np.allclose(proba, [[0.1742, 0.2188, 0.6070]], rtol=1e-3)
        assert np.allclose(margin, [0.0629], rtol=1e-3)

    def test_multiplicative_method_large_odds(self, large_odds):
        proba, margin = implied_odds.multiplicative_method(large_odds)
        assert np.allclose(proba, [0.54545455, 0.27272727, 0.18181818], rtol=1e-3)


class TestShinMethod:
    def test_shin_method(self, odds):
        proba, margin = implied_odds.shin_method(odds)
        assert np.allclose(proba, [0.1668, 0.2134, 0.6198], rtol=1e-3)
        assert np.isclose(margin, 0.0629, rtol=1e-3)

class TestImpliedProbsPower:
    def test_implied_probs_power(self, odds):
        proba, margin = implied_odds.power_method(odds)
        assert np.allclose(proba, [0.1645, 0.2099, 0.6256], rtol=1e-3)
        assert np.isclose(margin, 0.0629, rtol=1e-3)

    def test_implied_probs_power_non_convergence(self, odds):
        with pytest.raises(RuntimeError, match="Power root-finder did not converge."):
            implied_odds.power_method(odds, max_iter=1)

