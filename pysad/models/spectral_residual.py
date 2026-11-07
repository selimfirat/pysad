from __future__ import annotations

import numpy as np

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _int_at_least, check_univariate

# The amplitude below which a frequency counts as absent, and the smallest local average that a score divides by.
EPS = 1e-8


class SpectralResidual(BaseModel):
    """Scores each value of a univariate stream with the Spectral Residual (SR) saliency of the value in a sliding window :cite:`ren2019time`.

    For each value ``x_n``, the model takes the window of the last ``window_size`` values, which ends with ``x_n``, and appends ``num_estimated_points`` copies of the estimated value ``x_{n+1} = x_{n-m+1} + g m``. The gradient ``g`` is the average slope of the lines from ``x_{n-i}`` to ``x_n`` for ``i = 1, ..., m``. SR then transforms the extended window ``x``:

    1. The Fourier transform of ``x`` gives the log amplitude spectrum ``L(f)`` and the phase spectrum ``P(f)``.
    2. The spectral residual is ``R(f) = L(f) - AL(f)``, where ``AL(f)`` is the average of ``L`` over ``filter_size`` frequencies.
    3. The inverse Fourier transform gives the saliency map ``S(x) = |F^{-1}(exp(R(f) + i P(f)))|``.

    The score of ``x_n`` is ``(S(x_n) - S_bar) / S_bar``, where ``S_bar`` is the average saliency of the ``num_average_points`` values before ``x_n``. Higher scores are more anomalous. The paper labels a value as an anomaly if its score is above a threshold ``tau = 3``. The model returns the score itself, so you can apply the threshold to the scores. The score is 0.0 for the first ``window_size - 1`` values, until the window is full.

    Only the SR part of the paper is implemented. SR-CNN, which trains a convolutional network on the saliency map, is not.

    The implementation differs from the paper as follows. The authors' code at https://github.com/microsoft/anomalydetector settles the details that the paper leaves open:

    - The paper writes the average filter ``h_q`` as a ``q x q`` matrix for images. As in the authors' code, ``AL(f)`` is the trailing average of the last ``filter_size`` log amplitudes, over fewer amplitudes for the first frequencies.
    - The paper does not say what happens at a zero amplitude, where the log is undefined. As in the authors' code, a frequency with an amplitude of at most ``1e-8`` gets the log amplitude 0 and does not contribute to the saliency map.
    - The model divides by ``max(S_bar, 1e-8)``, so that the score stays finite on a constant series.

    On the same extended window, the saliency map of this model matches the authors' code to within 1e-14. The model differs from the authors' code where the code differs from the paper:

    - The code computes the estimated value from the 6 values before ``x_n``, so that ``x_n`` does not enter the gradient. The model follows Eqs. (8) and (9), whose gradient ends at ``x_n``. A spike at ``x_n`` then also raises the estimated values, which lowers its saliency. On a seasonal series with spikes, the estimate of the code gives an AUROC of about 0.97 and the estimate of the paper about 0.67.
    - The code scores ``|S(x_n) - S_avg| / S_avg / 10``, clipped to ``[0, 1]``, where ``S_avg`` is the average saliency of the last 40 values including ``x_n``. The model uses the score of Eq. (7) with ``z = 21``, as in the paper.
    - The code splits a series into batches and scores each batch in one window. The model scores each value in the window that ends with it, as in the streaming setting of the paper.

    Args:
        window_size (int): The sliding window size ``omega``, the number of real values in each window. Must be greater than ``num_gradient_points`` and ``num_average_points``. The paper uses 1440 on KPI, 64 on Yahoo and 30 on the Microsoft data (Default=64).
        filter_size (int): The width ``q`` of the average filter on the log amplitude spectrum (Default=3).
        num_average_points (int): The number ``z`` of preceding saliency values that ``S_bar`` averages (Default=21).
        num_estimated_points (int): The number ``kappa`` of estimated points that the model appends after ``x_n`` (Default=5).
        num_gradient_points (int): The number ``m`` of preceding values that the gradient of the estimated point uses (Default=5).

    Raises:
        ValueError: If a parameter is less than 1, or ``window_size`` is not greater than ``num_gradient_points`` and ``num_average_points``.
    """

    def __init__(
        self,
        window_size: int = 64,
        filter_size: int = 3,
        num_average_points: int = 21,
        num_estimated_points: int = 5,
        num_gradient_points: int = 5,
    ) -> None:
        self.filter_size = _int_at_least(filter_size, "filter_size", 1)
        self.num_average_points = _int_at_least(num_average_points, "num_average_points", 1)
        self.num_estimated_points = _int_at_least(num_estimated_points, "num_estimated_points", 1)
        self.num_gradient_points = _int_at_least(num_gradient_points, "num_gradient_points", 1)
        self.window_size = _int_at_least(
            value=window_size,
            name="window_size",
            minimum=max(self.num_gradient_points, self.num_average_points) + 1,
        )
        # Each window ends with the scored value, so the model keeps the window_size - 1 values before it.
        self.values_ = np.zeros(self.window_size - 1)
        self.num_values_ = 0

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> SpectralResidual:
        """Appends the next value to the window and drops the oldest value once the window is full.

        Args:
            X (float or np.float64 array): The instance to fit, holding exactly one value (e.g. a scalar or an array of shape (1,)).
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.

        Raises:
            ValueError: If `X` does not hold exactly one value.
        """
        value = check_univariate(X, "SpectralResidual")
        if self.num_values_ < self.values_.size:
            self.values_[self.num_values_] = value
            self.num_values_ += 1
        else:
            self.values_[:-1] = self.values_[1:]
            self.values_[-1] = value

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores `X` by its saliency in the window of the last ``window_size - 1`` fitted values followed by `X`. This method does not change the model.

        Args:
            X (float or np.float64 array): The instance to score, holding exactly one value (e.g. a scalar or an array of shape (1,)). Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The relative difference between the saliency of `X` and the average saliency of the preceding values, or 0.0 until the window is full.

        Raises:
            ValueError: If `X` does not hold exactly one value.
        """
        value = check_univariate(X, "SpectralResidual")
        if self.num_values_ < self.values_.size:
            return 0.0

        n = self.window_size
        m = self.num_gradient_points
        x = np.empty(n + self.num_estimated_points)
        x[: n - 1] = self.values_
        x[n - 1] = value
        # Eqs. (8) and (9): x_{n+1} = x_{n-m+1} + m * mean_i (x_n - x_{n-i}) / i.
        gradient = ((value - x[n - 1 - m : n - 1][::-1]) / np.arange(1, m + 1)).mean()
        x[n:] = x[n - m] + m * gradient

        spectrum = np.fft.fft(x)
        amplitude = np.abs(spectrum)
        present = amplitude > EPS
        log_amplitude = np.log(amplitude, out=np.zeros_like(amplitude), where=present)
        # The trailing average over the last filter_size log amplitudes, over fewer at the start.
        sums = log_amplitude.cumsum()
        sums[self.filter_size :] -= sums[: -self.filter_size].copy()
        average = sums / np.minimum(np.arange(1, x.size + 1), self.filter_size)
        # exp(R(f) + i P(f)) = exp(L(f) - AL(f)) * spectrum / A(f).
        residual = np.divide(spectrum, amplitude, out=np.zeros_like(spectrum), where=present)
        residual *= np.exp(log_amplitude - average)
        saliency = np.abs(np.fft.ifft(residual))

        preceding = saliency[n - 1 - self.num_average_points : n - 1].mean()
        return float((saliency[n - 1] - preceding) / max(preceding, EPS))
