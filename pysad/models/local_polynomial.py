from __future__ import annotations

import numpy as np

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _int_at_least, check_univariate
from pysad.utils.window import RingBuffer


class LocalPolynomialDetector(BaseModel):
    r"""Local polynomial fitting outlier and change detector :cite:`li2007unifying`.

    The detector fits a polynomial of a specified degree to a sliding window of past
    observations to predict the next value. The anomaly score is the squared
    prediction error :math:`(x_t - \hat{x}_t)^2`, where :math:`\hat{x}_t` is the
    predicted value based on the polynomial fitted to the previous window.

    For each new value :math:`x_t`, the model maintains a window of the last
    ``window_size`` observations. It fits a polynomial :math:`p` of degree
    ``degree`` to the pairs :math:`(x_{i}, x_{i+1})` in the window, then predicts
    :math:`\hat{x}_t = p(x_{t-1})`. The score is the squared error.

    The model keeps a fixed-size window, so its memory is bounded and one call costs
    O(window_size * degree^2) for the least-squares fit.
    It takes no ``random_state``, because it draws no random numbers.

    Deviations from the paper:

    - The paper defines both a forward predictor (past to present) and a backward
      predictor (future to present) and takes the maximum of both squared errors as
      the anomaly score. The backward predictor requires future data, which is not
      available in a streaming setting. This implementation uses only the forward
      predictor.
    - The paper normalizes the forward and backward scores by the variance of their
      respective neighborhoods. This implementation returns the raw squared prediction
      error. Postprocessors such as
      :class:`pysad.transform.postprocessing.NonparametricDynamicThresholdPostprocessor`
      can normalize scores if needed.

    Args:
        window_size (int): The number of past observations to use for fitting the
            polynomial. Must be at least 3 (Default=50).
        degree (int): The degree of the polynomial to fit. Must be at least 1
            (Default=1).

    Raises:
        ValueError: If ``window_size`` is less than 3, or ``degree`` is less than 1.
    """

    __bibtex__ = r"""
@InProceedings{li2007unifying,
  author    = {Li, Zhi and Ma, Hong and Mei, Yongbing},
  title     = {A Unifying Method for Outlier and Change Detection from Data Streams Based on Local Polynomial Fitting},
  booktitle = {Advances in Knowledge Discovery and Data Mining},
  year      = {2007},
  publisher = {Springer Berlin Heidelberg},
  address   = {Berlin, Heidelberg},
  pages     = {150--161},
  isbn      = {978-3-540-71701-0},
  doi       = {10.1007/978-3-540-71701-0_17}
}
"""

    _input_kind = "univariate"

    def __init__(self, window_size: int = 50, degree: int = 1) -> None:
        self.window_size = _int_at_least(window_size, "window_size", 3)
        self.degree = _int_at_least(degree, "degree", 1)

        self._window = RingBuffer(self.window_size)
        self._coeffs: np.ndarray | None = None

    def _fit_polynomial(self) -> None:
        """Fits a polynomial to the current window using least squares.

        The polynomial maps x_i to x_{i+1} for consecutive pairs in the window.
        """
        data = self._window.window_view()
        n = len(data)
        if n < self.degree + 2:
            self._coeffs = None
            return

        # Build design matrix for pairs (x_i, x_{i+1})
        # X = [x_0, x_1, ..., x_{n-2}], y = [x_1, x_2, ..., x_{n-1}]
        X = data[:-1]
        y = data[1:]

        # Design matrix for polynomial of degree `degree`
        # Using Horner's scheme: p(x) = c_0 + c_1*x + c_2*x^2 + ...
        A = np.column_stack([X**k for k in range(self.degree + 1)])

        # Least squares solve
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
        self._coeffs = coeffs

    def _predict(self, x: float) -> float:
        """Evaluates the fitted polynomial at x using Horner's method."""
        if self._coeffs is None:
            return x
        result = 0.0
        for c in reversed(self._coeffs):
            result = result * x + c
        return result

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> LocalPolynomialDetector:
        """Updates the window with the next value and re-fits the polynomial.

        Args:
            X (np.float64 array of shape (1,)): The value to fit. This model is univariate.
            y (int): The model ignores this label (Default=None).

        Returns:
            object: The model itself.
        """
        x = check_univariate(X, "LocalPolynomialDetector")
        self._window.append(x)
        self._fit_polynomial()
        return self

    def score_partial(self, X: np.ndarray) -> float:
        r"""Scores the value using the current polynomial fit, without changing the model.

        The score is the squared prediction error :math:`(x_t - \hat{x}_t)^2`,
        where :math:`\hat{x}_t` is predicted from the last value in the window.

        Args:
            X (np.float64 array of shape (1,)): The value to score. This model is univariate.

        Returns:
            float: The squared prediction error. Returns 0.0 if the window has fewer
                than degree + 2 points or the polynomial is not yet fitted.
        """
        x = check_univariate(X, "LocalPolynomialDetector")
        if self._window.count < self.degree + 2 or self._coeffs is None:
            return 0.0

        last_value = self._window.window_view()[-1]
        pred = self._predict(last_value)
        error = x - pred
        return float(error * error)
