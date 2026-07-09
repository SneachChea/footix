"""Tests for Bayesian confidence metrics based on posterior 1X2 samples."""

from __future__ import annotations

import numpy as np
import pytest

from footix.metrics import (
    confidence_1x2_from_samples,
    confidence_1x2_from_samples_array,
    confidence_curve,
)
from footix.utils.typing import SampleProbaResult


def test_confidence_is_near_zero_for_uniform_samples() -> None:
    """Uniform posterior probabilities should produce low confidence."""
    n_samples = 500
    p_samples = np.full((n_samples, 3), 1.0 / 3.0)

    components = confidence_1x2_from_samples_array(p_samples)

    assert components.confidence < 1e-8
    assert components.sharpness < 1e-8
    assert components.disagreement < 1e-8


def test_confidence_is_high_for_peaked_and_consistent_samples() -> None:
    """Peaked and consistent posterior probabilities should produce high confidence."""
    n_samples = 500
    p_samples = np.tile(np.array([0.95, 0.03, 0.02]), (n_samples, 1))

    components = confidence_1x2_from_samples_array(p_samples)

    assert components.confidence > 70.0
    assert components.sharpness > 0.7
    assert components.disagreement < 1e-8


def test_disagreement_penalizes_confidence_for_multi_modal_posterior() -> None:
    """Posterior disagreement should lower confidence even with a peaked mean distribution."""
    n_samples = 600
    half = n_samples // 2

    high_home = np.tile(np.array([0.90, 0.05, 0.05]), (half, 1))
    high_away = np.tile(np.array([0.05, 0.05, 0.90]), (n_samples - half, 1))
    p_samples = np.vstack((high_home, high_away))

    components_array = confidence_1x2_from_samples_array(p_samples)

    sample_result = SampleProbaResult(
        proba_home=p_samples[:, 0],
        proba_draw=p_samples[:, 1],
        proba_away=p_samples[:, 2],
    )
    components_namedtuple = confidence_1x2_from_samples(sample_result)

    assert components_array.disagreement > 0.4
    assert components_array.confidence < 60.0
    assert np.isclose(components_array.confidence, components_namedtuple.confidence)
    assert np.isclose(components_array.sharpness, components_namedtuple.sharpness)
    assert np.isclose(components_array.disagreement, components_namedtuple.disagreement)


def test_confidence_curve_preserves_endpoints_and_boosts_midrange() -> None:
    """Power-curve mapping should keep boundaries and boost with gamma < 1."""
    assert np.isclose(confidence_curve(0.0, gamma=0.7), 0.0)
    assert np.isclose(confidence_curve(100.0, gamma=0.7), 100.0)
    assert confidence_curve(25.0, gamma=0.7) > 25.0


def test_confidence_curve_is_monotone() -> None:
    """Rescaled confidence should preserve ordering for positive gamma."""
    low = confidence_curve(18.0, gamma=0.7)
    high = confidence_curve(42.0, gamma=0.7)
    assert low < high


def test_confidence_curve_rejects_non_positive_gamma() -> None:
    """Gamma must be strictly positive."""
    with pytest.raises(ValueError, match="gamma"):
        confidence_curve(20.0, gamma=0.0)


def test_confidence_from_samples_array_invalid_shape() -> None:
    """Invalid input shape raises ValueError."""
    with pytest.raises(ValueError, match="p_samples must have shape"):
        confidence_1x2_from_samples_array(np.ones((10, 4)))

    with pytest.raises(ValueError, match="p_samples must have shape"):
        confidence_1x2_from_samples_array(np.ones(10))
