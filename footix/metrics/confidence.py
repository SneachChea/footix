"""Confidence metrics derived from Bayesian posterior 1X2 samples.

This module provides utilities to convert posterior samples of match outcome
probabilities into a single confidence score in ``[0, 100]``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from footix.utils.typing import SampleProbaResult


class ConfidenceComponents(NamedTuple):
    """Decomposed confidence metrics for a 1X2 prediction.

    Attributes:
        confidence: Final confidence score in ``[0, 100]``.
        sharpness: Sharpness score in ``[0, 1]`` derived from normalized entropy.
        disagreement: Posterior disagreement score in ``[0, 1]`` derived from
            mutual information.
    """

    confidence: float
    sharpness: float
    disagreement: float


def confidence_curve(confidence: float, gamma: float = 0.7) -> float:
    """Rescale confidence with a monotone power curve.

    This helper is intended for readability in user interfaces while preserving
    the match ranking induced by the raw confidence score.

    The mapping is:
    ``c' = 100 * (clip(c, 0, 100) / 100) ** gamma``.

    Args:
        confidence: Raw confidence score.
        gamma: Positive exponent. Values below 1.0 boost mid-range scores,
            values above 1.0 compress them.

    Returns:
        Rescaled confidence in ``[0, 100]``.

    Raises:
        ValueError: If ``gamma`` is not strictly positive.
    """
    if gamma <= 0.0:
        raise ValueError("gamma must be strictly positive.")

    normalized = float(np.clip(confidence, 0.0, 100.0)) / 100.0
    return float(100.0 * (normalized**gamma))


def _entropy(probas: NDArray[np.float64], axis: int = -1) -> NDArray[np.float64]:
    """Compute Shannon entropy along a given axis.

    Args:
        probas: Probability array.
        axis: Axis used to compute entropy.

    Returns:
        Entropy values.
    """
    return -np.sum(probas * np.log(probas), axis=axis)


def confidence_1x2_from_samples_array(
    p_samples: NDArray[np.floating], eps: float = 1e-12
) -> ConfidenceComponents:
    """Compute confidence from posterior 1X2 probability samples.

    The score combines:
    - Sharpness: ``1 - H(mean_p) / log(3)``
    - Posterior disagreement: ``MI / log(3)`` where
      ``MI = H(mean_p) - E[H(p_s)]``

    Final score:
    ``confidence = 100 * sharpness * (1 - disagreement)``.

    Args:
        p_samples: Array with shape ``(n_samples, 3)`` containing posterior
            samples of ``[p_home, p_draw, p_away]``.
        eps: Numerical stability constant used for clipping.

    Returns:
        ConfidenceComponents with confidence in ``[0, 100]``.

    Raises:
        ValueError: If the input shape is invalid or no samples are provided.
    """
    samples = np.asarray(p_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 3:
        raise ValueError("p_samples must have shape (n_samples, 3).")
    if samples.shape[0] == 0:
        raise ValueError("At least one posterior sample is required.")

    clipped = np.clip(samples, eps, 1.0)
    row_sums = clipped.sum(axis=1, keepdims=True)
    normalized = clipped / row_sums

    mean_probs = normalized.mean(axis=0)
    entropy_mean = float(_entropy(mean_probs, axis=0))
    entropy_per_sample = _entropy(normalized, axis=1)
    expected_entropy = float(np.mean(entropy_per_sample))

    norm = float(np.log(3.0))
    sharpness = float(np.clip(1.0 - entropy_mean / norm, 0.0, 1.0))

    mutual_information = max(0.0, entropy_mean - expected_entropy)
    disagreement = float(np.clip(mutual_information / norm, 0.0, 1.0))

    confidence = float(np.clip(100.0 * 4.5 * sharpness * (1.0 - disagreement), 0.0, 100.0))
    return ConfidenceComponents(
        confidence=confidence,
        sharpness=sharpness,
        disagreement=disagreement,
    )


def confidence_1x2_from_samples(
    samples: SampleProbaResult,
    eps: float = 1e-12,
) -> ConfidenceComponents:
    """Compute confidence from a ``SampleProbaResult`` object.

    Args:
        samples: Posterior samples for home/draw/away outcome probabilities.
        eps: Numerical stability constant used for clipping.

    Returns:
        ConfidenceComponents with confidence in ``[0, 100]``.

    Raises:
        ValueError: If sample arrays have incompatible shapes.
    """
    proba_home = np.asarray(samples.proba_home, dtype=float)
    proba_draw = np.asarray(samples.proba_draw, dtype=float)
    proba_away = np.asarray(samples.proba_away, dtype=float)

    if not (proba_home.shape == proba_draw.shape == proba_away.shape):
        raise ValueError("Sample probability arrays must share the same shape.")

    stacked = np.column_stack((proba_home, proba_draw, proba_away))
    return confidence_1x2_from_samples_array(stacked, eps=eps)
