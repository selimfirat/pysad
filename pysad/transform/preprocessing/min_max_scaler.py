from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pysad.core.base_transformer import BaseTransformer
from pysad.statistics import MaxMeter, MinMeter


class MinMaxScaler(BaseTransformer):
    """Scale each feature with the running minimum and maximum.

    Keep one MinMeter and one MaxMeter per feature, created with keep_values=False so each
    meter keeps only its extreme value and memory stays constant as the stream grows.
    fit_partial updates both meters with the next instance, then transform_partial subtracts
    the running minimum and divides by the running range (maximum minus minimum).

    Only values inside the observed range land in [0, 1]. Scaling does not clip, so a value
    below the running minimum scales to a negative number and a value above the running
    maximum scales above one. For example, after fitting 0 and 2, transforming 4 returns 2 and
    transforming -1 returns -0.5. Callers that need a bounded output clip the result.

    The bounds only move when fit_partial is called. transform_partial never updates them, so
    transforming an unobserved value repeatedly keeps the same bounds and the same result.

    A constant feature has zero range, so every scaled value of that feature is zero.
    Before fit_partial is called, transform_partial returns the input unchanged because no
    statistics exist yet.

    fit_partial learns from the instance, then transform_partial scales it with the updated
    statistics. The first instance of each feature is both the minimum and the maximum, so its
    range is zero and it scales to zero.

    Attributes:
        minimums_ (list[MinMeter] or None): The running minimum of each feature. None until the first fit_partial call.
        maximums_ (list[MaxMeter] or None): The running maximum of each feature. None until the first fit_partial call.
    """

    def __init__(self) -> None:
        super().__init__(-1)
        self.minimums_: list[MinMeter] | None = None
        self.maximums_: list[MaxMeter] | None = None

    def fit_partial(self, X: NDArray[Any]) -> MinMaxScaler:
        """Update the running minimum and maximum of each feature with X.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            MinMaxScaler: The scaler.
        """
        if self.minimums_ is None:
            self._init_meters(X.size)

        minimums = self.minimums_
        maximums = self.maximums_
        assert minimums is not None and maximums is not None

        for minimum, maximum, x in zip(minimums, maximums, X, strict=True):
            value = float(x)
            minimum.update(value)
            maximum.update(value)

        return self

    def transform_partial(self, X: NDArray[Any]) -> NDArray[Any]:
        """Scale X with the running minimum and maximum of each feature.

        The result lies in [0, 1] only for values between the running minimum and maximum,
        because the output is not clipped. See the class docstring for an example. Bounds come
        from the last fit_partial call, so this method leaves them unchanged. Before the first
        fit_partial call, X is returned as a copy.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (num_features,): Scaled features. A value outside the
            running range scales below zero or above one, and a constant feature scales to zero.
        """
        if self.minimums_ is None:
            return X.astype(np.float64, copy=True)

        minimums = self.minimums_
        maximums = self.maximums_
        assert minimums is not None and maximums is not None

        scaled = np.empty(X.size, dtype=np.float64)
        for i, (x, minimum, maximum) in enumerate(zip(X, minimums, maximums, strict=True)):
            minimum_value = minimum.get()
            maximum_value = maximum.get()
            range_value = maximum_value - minimum_value
            if range_value == 0.0:
                scaled[i] = 0.0
            else:
                scaled[i] = (float(x) - minimum_value) / range_value

        return scaled

    def _init_meters(self, num_features: int) -> None:
        self.minimums_ = [MinMeter(keep_values=False) for _ in range(num_features)]
        self.maximums_ = [MaxMeter(keep_values=False) for _ in range(num_features)]
