from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from pysad.core.base_postprocessor import BasePostprocessor
from pysad.statistics.basic import AverageMeter, VarianceMeter
from pysad.statistics.running_statistic import RunningStatistic


class TSFMResidualAdapter(BasePostprocessor):
    """Normalizes forecasting residuals using rolling statistics for conformal calibration.

    This postprocessor transforms raw residuals (e.g., absolute prediction errors from a
    forecaster) into normalized scores using a rolling window of recent residuals. The
    normalized scores are better suited for :class:`~pysad.transform.probability_calibration.ConformalTSFMCalibrator`
    because they have more stable scale and distribution over time.

    The adapter maintains a rolling window of the most recent residuals and normalizes
    each new residual by subtracting the rolling mean and dividing by the rolling standard
    deviation. This is equivalent to a rolling Z-score normalization.

    The output is a continuous score where higher values mean more anomalous. The adapter
    can be chained with a conformal calibrator using the ``|`` operator:

    .. code-block:: python

        from pysad.integrations import ForecastingAnomalyDetector, TSFMResidualAdapter
        from pysad.transform.probability_calibration import ConformalTSFMCalibrator

        detector = ForecastingAnomalyDetector(forecaster)
        adapter = TSFMResidualAdapter(window_size=100)
        calibrator = ConformalTSFMCalibrator(alpha=0.1, gamma=0.01, window_size=200)

        # Chain: detector -> adapter -> calibrator
        pipeline = detector | adapter | calibrator

    Args:
        window_size (int): Size of the rolling window for mean and variance estimation.
            Must be at least 2 (Default=100).
        min_std (float): Minimum standard deviation to avoid division by zero.
            If the rolling standard deviation falls below this value, the output is 0.0
            (Default=1e-8).

    Raises:
        ValueError: If ``window_size`` is less than 2, or ``min_std`` is not positive.
    """

    _output_kind: ClassVar[Literal["score", "probability", "decision"]] = "score"

    def __init__(self, window_size: int = 100, min_std: float = 1e-8) -> None:
        if window_size < 2:
            raise ValueError(f"window_size must be at least 2, got {window_size}.")
        if min_std <= 0:
            raise ValueError(f"min_std must be positive, got {min_std}.")

        self.window_size = window_size
        self.min_std = float(min_std)

        # Running statistics for rolling mean and variance
        self._mean_meter = RunningStatistic(statistic_cls=AverageMeter, window_size=window_size)
        self._var_meter = RunningStatistic(statistic_cls=VarianceMeter, window_size=window_size)
        self.num_seen_ = 0

    def fit_partial(self, score: float) -> TSFMResidualAdapter:
        """Updates the rolling statistics with the next residual.

        Args:
            score (float): Input residual (e.g., absolute prediction error).

        Returns:
            object: self.
        """
        self._mean_meter.update(score)
        self._var_meter.update(score)
        self.num_seen_ += 1
        return self

    def transform_partial(self, score: float) -> float:
        """Normalizes the residual using the rolling statistics.

        Args:
            score (float): Input residual.

        Returns:
            float: Normalized residual, or 0.0 if not enough data or variance is too small.
        """
        if self.num_seen_ < 2:
            return 0.0

        mean = self._mean_meter.statistic.get()
        variance = self._var_meter.statistic.get()

        if variance < self.min_std**2:
            return 0.0

        return float((score - mean) / np.sqrt(variance))

    def transform(self, scores: NDArray[Any]) -> NDArray[Any]:
        """Normalizes a batch of residuals using the current frozen statistics."""
        arr = np.asarray(scores, dtype=np.float64)
        if arr.ndim != 1:
            return super().transform(scores)
        n = len(arr)
        if n == 0 or self.num_seen_ < 2:
            return np.zeros(n, dtype=np.float64)
        mean = self._mean_meter.statistic.get()
        variance = self._var_meter.statistic.get()
        if variance < self.min_std**2:
            return np.zeros(n, dtype=np.float64)
        return np.asarray((arr - mean) / np.sqrt(variance), dtype=np.float64)
