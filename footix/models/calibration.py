"""Out-of-sample 1X2 outcome calibration.

Learns the transform ``softmax(tau * log p + bias)`` from past model
predictions and their realized outcomes only. The model itself is never
retrained here and no future information is used, so the calibrated
probabilities are causal by construction. With ``tau = 1`` and ``bias = 0``
the transform is the exact identity, which is also the default until
``warmup`` match outcomes have been accumulated.

The same ``apply`` call is used for point probabilities (``(3,)`` or
``(n, 3)``) and for posterior draws (``(n_draws, n_outcomes)``), keeping
point and sampled 1X2 probabilities consistent.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

EPS = 1e-12


class OutcomeCalibrator:
    """Temperature-and-bias 1X2 calibrator, fitted only on past outcomes.

    Attributes:
        tau: Temperature; 1.0 is the identity.
        bias: Class-wise logit bias, shape (3,); zeros are the identity.

    """

    def __init__(self, warmup: int = 50, reg: float = 1e-2) -> None:
        self.warmup = warmup
        self.reg = reg
        self.tau = 1.0
        self.bias = np.zeros(3, dtype=float)
        self._probs: list[np.ndarray] = []
        self._outcomes: list[np.ndarray] = []

    def accumulate(self, probs: np.ndarray, outcomes: np.ndarray) -> None:
        """Store one batch of raw predictions and realized 1X2 outcomes.

        Args:
            probs: Raw model probabilities, shape ``(3,)`` or ``(n, 3)``.
            outcomes: Realized outcome index (0 = H, 1 = D, 2 = A), shape
                ``(n,)``.
        """
        probs = np.atleast_2d(np.asarray(probs, dtype=float))
        outcomes = np.asarray(outcomes, dtype=int).reshape(-1)
        if probs.shape[0] != outcomes.size:
            raise ValueError("outcomes must match the number of probability rows")
        self._probs.append(probs)
        self._outcomes.append(outcomes)

    def fit(self) -> None:
        """Fit on accumulated past data; no-op before ``warmup`` observations."""
        n_obs = sum(len(outcomes) for outcomes in self._outcomes)
        if n_obs < self.warmup:
            self.tau, self.bias = 1.0, np.zeros(3)
            return
        probs = np.vstack(self._probs)
        outcomes = np.concatenate(self._outcomes)
        logp = np.log(np.clip(probs, EPS, 1.0))

        def objective(theta: np.ndarray) -> float:
            log_tau, *bias = theta
            logits = np.exp(log_tau) * logp + np.asarray(bias)
            logits -= logits.max(axis=-1, keepdims=True)
            softmax = np.exp(logits)
            softmax /= softmax.sum(axis=-1, keepdims=True)
            nll = -np.log(np.maximum(softmax[np.arange(outcomes.size), outcomes], EPS)).sum()
            return nll + self.reg * (log_tau**2 + float(np.dot(bias, bias)))

        result = minimize(objective, np.zeros(4), method="BFGS")
        log_tau, *bias = result.x
        self.tau = float(np.exp(log_tau))
        self.bias = np.asarray(bias)

    def apply(self, probs: np.ndarray) -> np.ndarray:
        """Apply the learned transform to point or draw probabilities.

        Args:
            probs: H/D/A probabilities, shape ``(3,)`` or ``(..., 3)``.

        Returns:
            Calibrated probabilities, same shape as ``probs``.
        """
        probs = np.asarray(probs, dtype=float)
        logits = self.tau * np.log(np.clip(probs, EPS, 1.0)) + self.bias
        logits -= logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=-1, keepdims=True)
