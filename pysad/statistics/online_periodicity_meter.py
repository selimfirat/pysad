from __future__ import annotations

from collections import deque

import numpy as np

from pysad.core.base_statistic import UnivariateStatistic


class OnlinePeriodicityMeter(UnivariateStatistic):
    """Estimates the period of a univariate stream from its online autocorrelation.

    The meter keeps an exponentially weighted moving average (EWMA) of the mean, the second
    moment, and, for every lag up to ``max_lag``, the product of the current value with the value
    that many steps earlier. From these it forms the autocorrelation at each lag and keeps the lag
    with the largest autocorrelation as the estimated period. A ring buffer of the last
    ``max_lag`` values provides the lagged values, so memory stays bounded and the cost of one
    update is independent of the stream length.

    The autocorrelation at lag k is ``(c[k] - mean^2) / variance``, where c[k] is the EWMA of
    ``x[t] * x[t-k]`` and mean and variance are the EWMA mean and variance of ``x``. The first
    product observed for a lag seeds its EWMA. The estimated period is the lag with the largest
    positive autocorrelation, or 0 when no lag has a positive autocorrelation. The estimate is a
    heuristic and is unreliable until the meter has seen many points.

    This applies the exponentially weighted moving average estimator of
    :cite:`roberts1959control` to the mean, the second moment and the lagged cross-products that
    define the autocorrelation.

    Args:
        max_lag (int): The largest lag tested. The estimated period is at most ``max_lag``. It
            must be at least 1 (Default=100).
        alpha (float): The EWMA weight given to each new observation, in the interval (0, 1].
            Larger values forget older observations faster (Default=0.01).

    Raises:
        ValueError: If ``max_lag`` is less than 1 or ``alpha`` is not in the interval (0, 1].

    Attributes:
        period (float): The estimated period, the lag with the largest autocorrelation. It is
            0.0 until the meter has seen at least two values and some lag has a positive
            autocorrelation.
    """

    def __init__(self, max_lag: int = 100, alpha: float = 0.01) -> None:
        if max_lag < 1:
            raise ValueError("max_lag must be at least 1.")

        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the interval (0, 1].")

        self.max_lag = max_lag
        self.alpha = alpha

        self.count = 0
        self.mean = 0.0
        self.second_moment = 0.0
        self.variance = 0.0

        # The buffer holds x[t-max_lag] ... x[t], so lag max_lag is computable once count > max_lag.
        self.buffer: deque[float] = deque(maxlen=max_lag + 1)

        # cross[j] holds the EWMA of x[t] * x[t-(j+1)]. cross_seen[j] marks the first observation,
        # which seeds the EWMA instead of starting from zero.
        self.cross = np.zeros(max_lag, dtype=np.float64)
        self.cross_seen = np.zeros(max_lag, dtype=bool)

        self.period = 0.0

    def update(self, num: float) -> OnlinePeriodicityMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        self.count += 1
        self.buffer.append(num)

        one_minus_alpha = 1.0 - self.alpha
        if self.count == 1:
            self.mean = num
            self.second_moment = num * num
        else:
            self.mean = self.alpha * num + one_minus_alpha * self.mean
            self.second_moment = self.alpha * (num * num) + one_minus_alpha * self.second_moment

        self.variance = self.second_moment - self.mean * self.mean
        # Rounding can make the variance slightly negative for a nearly constant stream.
        if self.variance < 0.0:
            self.variance = 0.0

        # Autocorrelation at lag k needs x[t-k], which is k+1 positions from the end of the buffer
        # (the last value is x[t]). It is available once count > k.
        num_lags = min(self.max_lag, self.count - 1)
        best_period = 0.0
        best_autocorrelation = 0.0
        for k in range(1, num_lags + 1):
            j = k - 1
            lagged = self.buffer[-(k + 1)]
            product = num * lagged
            if not self.cross_seen[j]:
                self.cross[j] = product
                self.cross_seen[j] = True
            else:
                self.cross[j] = self.alpha * product + one_minus_alpha * self.cross[j]

            if self.variance > 0.0:
                autocorrelation = (self.cross[j] - self.mean * self.mean) / self.variance
                if autocorrelation > 1.0:
                    autocorrelation = 1.0
                elif autocorrelation < -1.0:
                    autocorrelation = -1.0
            else:
                autocorrelation = 0.0

            if autocorrelation > best_autocorrelation:
                best_autocorrelation = autocorrelation
                best_period = float(k)

        self.period = best_period
        return self

    def remove(self, num: float) -> OnlinePeriodicityMeter:
        """Raises ``NotImplementedError``, because the meter cannot remove observations.

        The meter keeps exponentially weighted summaries and a bounded ring buffer, so a past
        observation cannot be taken back. Use a windowed batch average for windows.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        raise NotImplementedError(
            "The meter keeps exponentially weighted summaries, so it cannot remove observations."
        )

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The estimated period, the lag with the largest autocorrelation. It is 0.0
                until the meter has seen at least two values and some lag has a positive
                autocorrelation.
        """
        return self.period
