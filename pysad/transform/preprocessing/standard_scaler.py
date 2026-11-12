from __future__ import annotations

import math

import numpy as np

from pysad.core.base_transformer import BaseTransformer
from pysad.statistics import AverageMeter, VarianceMeter


class StandardScaler(BaseTransformer):
    """Scale each feature to zero mean and unit variance with running statistics.

    Keep one AverageMeter and one VarianceMeter per feature, so memory stays constant as the
    stream grows: it grows with the number of features, not with the number of instances.
    fit_partial updates both meters with the next instance, then transform_partial subtracts
    the running mean and divides by the running standard deviation.

    A constant feature has zero variance, so its running standard deviation is zero and every
    scaled value of that feature is zero. Before fit_partial is called, transform_partial
    returns the input unchanged because no statistics exist yet.

    fit_partial learns from the instance, then transform_partial scales it with the updated
    statistics. The first instance of each feature has zero variance, so it scales to zero.

    Attributes:
        means_ (list[AverageMeter] or None): The running mean of each feature. None until the first fit_partial call.
        variances_ (list[VarianceMeter] or None): The running variance of each feature. None until the first fit_partial call.
    """

    def __init__(self) -> None:
        super().__init__(-1)
        self.means_: list[AverageMeter] | None = None
        self.variances_: list[VarianceMeter] | None = None

    def fit_partial(self, X: np.ndarray) -> StandardScaler:
        """Update the running mean and variance of each feature with X.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            StandardScaler: The scaler.
        """
        if self.means_ is None:
            self._init_meters(X.size)

        means = self.means_
        variances = self.variances_
        assert means is not None and variances is not None

        for mean, x in zip(means, X, strict=True):
            mean.update(float(x))

        for variance, x in zip(variances, X, strict=True):
            variance.update(float(x))

        return self

    def transform_partial(self, X: np.ndarray) -> np.ndarray:
        """Scale X with the running mean and standard deviation of each feature.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (num_features,): Scaled features. A constant feature scales to zero.
        """
        if self.means_ is None:
            return X.astype(np.float64, copy=True)

        means = self.means_
        variances = self.variances_
        assert means is not None and variances is not None

        scaled = np.empty(X.size, dtype=np.float64)
        for i, (x, mean, variance) in enumerate(zip(X, means, variances, strict=True)):
            mean_value = mean.get()
            variance_value = variance.get()
            if variance_value == 0.0:
                scaled[i] = 0.0
            else:
                scaled[i] = (float(x) - mean_value) / math.sqrt(variance_value)

        return scaled

    def _init_meters(self, num_features: int) -> None:
        self.means_ = [AverageMeter() for _ in range(num_features)]
        self.variances_ = [VarianceMeter() for _ in range(num_features)]
