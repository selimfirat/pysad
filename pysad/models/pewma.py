from __future__ import annotations

import math

import numpy as np

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _finite_float, _int_at_least, check_univariate


class PEWMA(BaseModel):
    r"""Probabilistic exponentially weighted moving average (PEWMA) :cite:`carter2012probabilistic`.

    PEWMA keeps exponentially weighted moving averages :math:`s_1` of the values and :math:`s_2` of their squares, with the estimates :math:`\hat{X}_t = s_1` and :math:`\hat{\sigma}_t = \sqrt{s_2 - s_1^2}`. For each value :math:`X_t` it computes :math:`Z_t = (X_t - \hat{X}_t) / \hat{\sigma}_t` and :math:`P_t = e^{-Z_t^2 / 2} / \sqrt{2 \pi}`, then updates both averages with the weight

    .. math::

        \alpha_t = 1 - 1/t \text{ for } t \le T, \qquad \alpha_t = (1 - \beta P_t) \alpha \text{ for } t > T,

    as in :math:`s_1 \leftarrow \alpha_t s_1 + (1 - \alpha_t) X_t`. During the training period of :math:`T` values, :math:`s_1` and :math:`s_2` are the exact means of the values and of their squares. After it, a likely value gets a small weight :math:`\alpha_t` and moves the estimates more than an unlikely one. With ``beta=0``, the model is an EWMA with the weight :math:`\alpha`. The model keeps two numbers, so its memory is fixed and one call costs O(1). It takes no ``random_state``, because it draws no random numbers.

    The score of a value is :math:`|Z_t|`, computed with the estimates before the model fits the value.

    Deviations from the paper:

    - The paper raises an alert when :math:`P_t` is below a threshold. The model returns :math:`|Z_t|`, which ranks values in the same order as :math:`-P_t`, so that a higher score is more anomalous. :math:`P_t` is below a threshold :math:`\tau` exactly when :math:`|Z_t|` is above :math:`\sqrt{-2 \ln(\tau \sqrt{2 \pi})}`.
    - The paper does not define :math:`Z_t` when :math:`\hat{\sigma}_t = 0`, as before the second value or on a constant stream. The model then uses :math:`Z_t = 0`, so the value scores 0 and gets the largest probability :math:`P_t`. Rounding can make :math:`s_2 - s_1^2` slightly negative, and the model then uses :math:`\hat{\sigma}_t = 0`.

    Args:
        alpha (float): The weight :math:`\alpha` of the previous averages after the training period, in [0, 1] (Default=0.99).
        beta (float): The weight :math:`\beta` of the probability of a value in :math:`\alpha_t`, in [0, 1]. With 0, the model is an EWMA (Default=1.0).
        training_period (int): The number :math:`T` of first values whose averages are exact means. It must be at least 1 (Default=14).

    Raises:
        ValueError: If ``alpha`` or ``beta`` is not in [0, 1], or ``training_period`` is less than 1.
    """

    def __init__(self, alpha: float = 0.99, beta: float = 1.0, training_period: int = 14) -> None:
        self.alpha = _finite_float(alpha, "alpha")
        self.beta = _finite_float(beta, "beta")
        for name, value in (("alpha", self.alpha), ("beta", self.beta)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        self.training_period = _int_at_least(training_period, "training_period", 1)

        self.count_ = 0
        self.s1_ = 0.0
        self.s2_ = 0.0

    def _z(self, x: float) -> float:
        """The z-score of ``x`` with the current estimates, 0 if the standard deviation is 0."""
        variance = self.s2_ - self.s1_**2
        return (x - self.s1_) / math.sqrt(variance) if variance > 0.0 else 0.0

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> PEWMA:
        """Updates the averages with the next value.

        Args:
            X (np.float64 array of shape (1,)): The value to fit. This model is univariate.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        x = check_univariate(X, "PEWMA")
        self.count_ += 1
        if self.count_ <= self.training_period:
            weight = 1.0 - 1.0 / self.count_
        else:
            p = math.exp(-(self._z(x) ** 2) / 2.0) / math.sqrt(2.0 * math.pi)
            weight = (1.0 - self.beta * p) * self.alpha

        self.s1_ = weight * self.s1_ + (1.0 - weight) * x
        self.s2_ = weight * self.s2_ + (1.0 - weight) * x**2

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the value with the current estimates, without changing the model.

        Args:
            X (np.float64 array of shape (1,)): The value to score. This model is univariate.

        Returns:
            float: The absolute z-score :math:`|Z_t|` of the value. It is 0 while the standard deviation is 0.
        """
        return abs(self._z(check_univariate(X, "PEWMA")))
