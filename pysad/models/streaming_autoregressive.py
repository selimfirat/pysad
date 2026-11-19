from __future__ import annotations

from collections import deque

import numpy as np
from scipy.linalg import toeplitz

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _finite_float, _int_at_least, check_univariate


class _StreamingAR:
    """A univariate autoregressive model learned online with the SDAR algorithm :cite:`takeuchi2006unifying`.

    The model maintains an exponentially discounted mean, autocovariances, and AR coefficients.
    For each value ``x_t``, it updates the mean ``mu = (1 - r) mu + r x_t`` and the
    autocovariances ``C_j = (1 - r) C_j + r (x_t - mu)(x_{t-j} - mu)`` for ``j = 0, ..., k``,
    solves the Yule-Walker equations for the AR coefficients ``omega``, and updates the
    variance ``sigma^2 = (1 - r) sigma^2 + r (x_t - x_hat_t)^2`` with the prediction
    ``x_hat_t = sum_i omega_i (x_{t-i} - mu) + mu``.
    """

    def __init__(self, order: int, discount: float) -> None:
        self.discount = discount
        self.history: deque[float] = deque(maxlen=order)  # The last values, oldest first.
        self.mean = 0.0
        self.autocovariances = np.zeros(order + 1)  # C_0, ..., C_k
        self.coefficients = np.zeros(order)  # omega_1, ..., omega_k
        self.variance = 1.0

    def _prediction(self) -> float:
        """Returns the one-step forecast of the next value."""
        past = np.array(self.history)[::-1]  # x_{t-1}, ..., x_{t-k}
        return float(self.coefficients @ (past - self.mean)) + self.mean

    def residual(self, x: float) -> float | None:
        """Returns the forecasting residual ``x - x_hat`` of ``x`` under the current model, or None until the model holds ``order`` values."""
        if len(self.history) < len(self.coefficients):
            return None
        return x - self._prediction()

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


class StreamingAutoregressiveDetector(BaseModel):
    """Scores each value of a univariate stream by its forecasting residual from a streaming autoregressive model :cite:`takeuchi2006unifying`.

    The model learns an autoregressive (AR) model of order ``k`` with the sequentially discounting AR (SDAR) algorithm.
    For each value ``x_t``, SDAR updates the mean ``mu = (1 - r) mu + r x_t`` and the
    autocovariances ``C_j = (1 - r) C_j + r (x_t - mu)(x_{t-j} - mu)`` for ``j = 0, ..., k``,
    solves the Yule-Walker equations for the AR coefficients ``omega``, and updates the
    variance ``sigma^2 = (1 - r) sigma^2 + r (x_t - x_hat_t)^2`` with the prediction
    ``x_hat_t = sum_i omega_i (x_{t-i} - mu) + mu``. A larger ``discount`` ``r`` forgets the past faster.

    The anomaly score of ``x_t`` is the absolute forecasting residual ``|x_t - x_hat_t|`` computed
    before the model learns ``x_t``. Higher scores are more anomalous. The score is 0.0 for the first
    ``order`` values until the model has enough history to forecast.

    The implementation differs from the paper as follows:

    - The model takes a univariate series. The paper also defines SDAR for vectors.
    - The paper does not give the initial parameters. The model starts with ``mu = 0``,
      ``C_j = 0``, ``omega = 0`` and ``sigma^2 = 1``, which fits a series of about unit scale,
      such as a min-max scaled series.
    - The model solves the Yule-Walker equations with least squares, so that it gets coefficients
      when the autocovariance matrix is singular, for example on a constant series.
    - The paper's outlier score is the logarithmic loss ``-log p(x_t)``. This model returns the
      absolute residual ``|x_t - x_hat_t|``, which ranks values in the same order as the loss under
      a Gaussian assumption with constant variance. The loss and the residual differ by a monotonic
      transformation when the variance is constant; here the variance is also tracked but the score
      uses the simpler residual for interpretability.
    - The scores are 0.0 during the warm-up. The paper fills the warm-up with dummy values instead.

    Args:
        order (int): The order ``k`` of the AR model (Default=1).
        discount (float): The discounting rate ``r`` of the AR model, in ``(0, 1)``. The weight of a
            value decays by ``1 - r`` with each later value (Default=0.01).

    Raises:
        ValueError: If ``discount`` is not in ``(0, 1)``, or ``order`` is less than 1.
    """

    def __init__(
        self,
        order: int = 1,
        discount: float = 0.01,
    ) -> None:
        self.order = _int_at_least(order, "order", 1)
        self.discount = _finite_float(discount, "discount")
        if not 0.0 < self.discount < 1.0:
            raise ValueError(f"discount must be in (0, 1), got {self.discount}.")

        self._model = _StreamingAR(self.order, self.discount)

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> StreamingAutoregressiveDetector:
        """Updates the AR model with the next value.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        x = check_univariate(X, "StreamingAutoregressiveDetector")
        self._model.update(x)
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the next value by its absolute forecasting residual under the current AR model.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value. Higher scores represent more anomalous values.

        Returns:
            float: The absolute forecasting residual of ``X``, or 0.0 until the model has enough history.
        """
        x = check_univariate(X, "StreamingAutoregressiveDetector")
        residual = self._model.residual(x)
        return 0.0 if residual is None else abs(residual)
