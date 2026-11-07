from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy.linalg import toeplitz

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _finite_float, _int_at_least, check_univariate

# Added to the variance when the model computes a density, so that the density stays finite on a constant series.
VARIANCE_REGULARIZATION = 1e-6


class _SDAR:
    """A univariate autoregressive model that the SDAR algorithm learns one value at a time."""

    def __init__(self, order: int, discount: float) -> None:
        self.discount = discount
        self.history: deque[float] = deque(maxlen=order)  # The last values, oldest first.
        self.mean = 0.0
        self.autocovariances = np.zeros(order + 1)  # C_0, ..., C_k
        self.coefficients = np.zeros(order)  # omega_1, ..., omega_k
        self.variance = 1.0

    def _prediction(self) -> float:
        past = np.array(self.history)[::-1]  # x_{t-1}, ..., x_{t-k}
        return float(self.coefficients @ (past - self.mean)) + self.mean

    def log_loss(self, x: float) -> float | None:
        """Returns ``-log p(x)`` under the current model, or None until the model holds ``order`` values."""
        if len(self.history) < len(self.coefficients):
            return None

        variance = self.variance + VARIANCE_REGULARIZATION
        return 0.5 * (math.log(2.0 * math.pi * variance) + (x - self._prediction()) ** 2 / variance)

    def update(self, x: float) -> None:
        """One SDAR step on ``x``. The model only stores ``x`` until it holds ``order`` values."""
        if len(self.history) == len(self.coefficients):
            r = self.discount
            self.mean = (1.0 - r) * self.mean + r * x
            lagged = np.array([x, *reversed(self.history)]) - self.mean  # x_t, ..., x_{t-k}
            self.autocovariances = (1.0 - r) * self.autocovariances + r * lagged[0] * lagged
            # The Yule-Walker equations. Least squares gives a solution when the matrix is singular.
            self.coefficients = np.linalg.lstsq(
                toeplitz(self.autocovariances[:-1]), self.autocovariances[1:], rcond=None
            )[0]
            self.variance = (1.0 - r) * self.variance + r * (x - self._prediction()) ** 2

        self.history.append(x)


class ChangeFinder(BaseModel):
    """Scores outliers and change points in a univariate stream with two autoregressive models that the SDAR algorithm learns online :cite:`takeuchi2006unifying`.

    The model learns an autoregressive (AR) model of order ``k`` with the sequentially discounting AR (SDAR) algorithm. For each value ``x_t``, SDAR updates the mean ``mu = (1 - r) mu + r x_t`` and the autocovariances ``C_j = (1 - r) C_j + r (x_t - mu)(x_{t-j} - mu)`` for ``j = 0, ..., k``, solves the Yule-Walker equations for the AR coefficients ``omega``, and updates the variance ``sigma^2 = (1 - r) sigma^2 + r (x_t - x_hat_t)^2`` with the prediction ``x_hat_t = sum_i omega_i (x_{t-i} - mu) + mu``. A larger ``discount`` ``r`` forgets the past faster.

    ChangeFinder runs in two stages:

    1. The outlier score of ``x_t`` is its logarithmic loss ``-log p(x_t | x_{t-1}, ..., x_{t-k})`` under the first AR model before it learns ``x_t``.
    2. The moving average ``y_t`` of the last ``window_size`` outlier scores goes to a second AR model. The change point score of ``x_t`` is the moving average of the last ``window_size`` logarithmic losses of ``y`` under the second model.

    Higher scores are more anomalous. The outlier score is 0.0 for the first ``order`` values, and the change point score is 0.0 for the first ``2 * (order + window_size - 1)`` values.

    The implementation differs from the paper as follows:

    - The model takes a univariate series. The paper also defines SDAR for vectors.
    - The paper does not give the initial parameters. Each AR model starts with ``mu = 0``, ``C_j = 0``, ``omega = 0`` and ``sigma^2 = 1``, which fits a series of about unit scale, such as a min-max scaled series.
    - An AR model only stores the first ``order`` values that it gets, because SDAR needs ``order`` past values to learn and to score a value.
    - The model solves the Yule-Walker equations with least squares, so that it gets coefficients when the autocovariance matrix is singular, for example on a constant series.
    - The model adds ``1e-6`` to the variance when it computes a density, so that the logarithmic loss stays finite on a constant series.
    - Both AR models share one ``order`` and one ``discount``. The paper lets the two models use different orders, and its experiments use 2 and 3.
    - The default ``discount=0.01`` differs from the ``r = 0.02`` of the paper's experiments.
    - The scores are 0.0 during the warm-up. Eq. (3) of the paper fills the warm-up with dummy values instead.

    Args:
        order (int): The order ``k`` of both AR models (Default=1).
        discount (float): The discounting rate ``r`` of both AR models, in ``(0, 1)``. The weight of a value decays by ``1 - r`` with each later value (Default=0.01).
        window_size (int): The width ``T`` of both moving averages (Default=5).
        change_point (bool): Whether to return the change point score. If False, the model returns the outlier score (Default=True).

    Raises:
        ValueError: If ``discount`` is not in ``(0, 1)``, or ``order`` or ``window_size`` is less than 1.
    """

    def __init__(
        self,
        order: int = 1,
        discount: float = 0.01,
        window_size: int = 5,
        change_point: bool = True,
    ) -> None:
        self.order = _int_at_least(order, "order", 1)
        self.discount = _finite_float(discount, "discount")
        if not 0.0 < self.discount < 1.0:
            raise ValueError(f"discount must be in (0, 1), got {self.discount}.")
        self.window_size = _int_at_least(window_size, "window_size", 1)
        self.change_point = change_point
        self._outlier_model = _SDAR(self.order, self.discount)
        self._change_model = _SDAR(self.order, self.discount)
        # The last window_size - 1 scores of each stage, which a moving average adds to the newest score.
        self._outlier_scores: deque[float] = deque(maxlen=self.window_size - 1)
        self._change_scores: deque[float] = deque(maxlen=self.window_size - 1)

    def _losses(self, x: float) -> tuple[float | None, float | None, float | None]:
        """Returns the outlier score of ``x``, its moving average and the log loss of the average under the second model, each None until it is defined."""
        outlier = self._outlier_model.log_loss(x)
        if outlier is None or len(self._outlier_scores) < self.window_size - 1:
            return outlier, None, None

        smoothed = (sum(self._outlier_scores) + outlier) / self.window_size
        return outlier, smoothed, self._change_model.log_loss(smoothed)

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> ChangeFinder:
        """Updates both AR models with the next value.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        x = check_univariate(X, "ChangeFinder")
        outlier, smoothed, change = self._losses(x)
        self._outlier_model.update(x)
        if outlier is not None:
            self._outlier_scores.append(outlier)
        if smoothed is not None:
            self._change_model.update(smoothed)
        if change is not None:
            self._change_scores.append(change)
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the next value under the current AR models.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value. Higher scores represent more anomalous values.

        Returns:
            float: The change point score of ``X``, or its outlier score if ``change_point`` is False.
        """
        outlier, _, change = self._losses(check_univariate(X, "ChangeFinder"))
        if not self.change_point:
            return 0.0 if outlier is None else outlier
        if change is None or len(self._change_scores) < self.window_size - 1:
            return 0.0
        return (sum(self._change_scores) + change) / self.window_size
