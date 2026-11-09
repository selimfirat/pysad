from __future__ import annotations

import numpy as np

from pysad.core.base_postprocessor import BasePostprocessor
from pysad.statistics.average_meter import AverageMeter
from pysad.statistics.max_meter import MaxMeter
from pysad.statistics.median_meter import MedianMeter
from pysad.statistics.running_statistic import RunningStatistic
from pysad.statistics.variance_meter import VarianceMeter
from pysad.transform.postprocessing.postprocessors import _MeterPostprocessor, _ZScorePostprocessor


class RunningAveragePostprocessor(_MeterPostprocessor):
    """A postprocessor that converts a score to the average of all previous scores in the window.

    Args:
        window_size (int): Length of the window
    """

    def __init__(self, window_size: int) -> None:
        super().__init__(RunningStatistic(statistic_cls=AverageMeter, window_size=window_size))


class RunningMaxPostprocessor(_MeterPostprocessor):
    """A postprocessor that converts a score to the maximum of all previous scores in the window.

    Args:
        window_size (int): Length of the window
    """

    def __init__(self, window_size: int) -> None:
        super().__init__(RunningStatistic(statistic_cls=MaxMeter, window_size=window_size))


class RunningMedianPostprocessor(_MeterPostprocessor):
    """A postprocessor that converts a score to the median of all previous scores in the window.

    Args:
        window_size (int): Length of the window
    """

    def __init__(self, window_size: int) -> None:
        super().__init__(RunningStatistic(statistic_cls=MedianMeter, window_size=window_size))


class RunningZScorePostprocessor(_ZScorePostprocessor):
    """A postprocessor that normalizes score using Z-score normalization with the statistics of the window.

    Args:
        window_size (int): Length of the window
    """

    def __init__(self, window_size: int) -> None:
        super().__init__(
            RunningStatistic(statistic_cls=VarianceMeter, window_size=window_size),
            RunningStatistic(statistic_cls=AverageMeter, window_size=window_size),
        )


_Z_VALUES = tuple(z / 2 for z in range(5, 21))


class NonparametricDynamicThresholdPostprocessor(BasePostprocessor):
    """The nonparametric dynamic threshold of Telemanom :cite:`hundman2018detecting`.

    The postprocessor smooths the scores with an exponentially weighted moving average (EWMA) and keeps the last ``window_size`` smoothed scores. For each candidate ``z``, it sets the threshold ``mean + z * std`` of the window (Eq. 2). It picks the threshold that maximizes the percent decrease of the mean and the standard deviation of the window when the scores above the threshold are removed, divided by ``num_anomalies + num_sequences ** 2`` (Eq. 3). Then it prunes the anomalous sequences of the window: it sorts the maximum of each sequence in descending order, appends the maximum score below the threshold, and keeps only the sequences before the last step whose percent decrease exceeds ``p``. If the current smoothed score is in a kept sequence, the output is the score of that sequence, ``(max(sequence) - threshold) / (mean + std)`` (Eq. 4). Otherwise, the output is 0.

    The scores must be nonnegative, for example the prediction errors of a forecaster, because the method divides by the mean and by the maximum of a sequence.

    These are the differences from the paper and from its reference implementation (https://github.com/khundman/telemanom, BSD-style Caltech license):

    - The paper and the reference evaluate a batch of 70 values at a time. This class evaluates each score once, when it arrives, so that it gives one output per score.
    - The window is shorter than ``window_size`` at the start of the stream. A window of fewer than 9 scores cannot exceed ``mean + 2.5 * std`` and gives 0.
    - The EWMA is the bias-corrected form of ``pandas.DataFrame.ewm(span=span)`` that the reference uses. The paper does not give the span, and the default 105 is the value of the reference configuration (5% of 30 batches of 70 values).
    - The default ``z_values`` start at 2.5 and step by 0.5, as in the reference. The upper bound 10 comes from the paper, which reports that values between two and ten work well. The reference goes up to 11.5 (``np.arange(2.5, 12.0, 0.5)``).
    - The reference also thresholds the errors flipped around their mean, extends each anomaly by 100 values, skips errors that are small compared with the range of the raw telemetry, limits each window to 5 sequences and to less than half of the window, and skips the first values of the stream. The paper describes none of these steps, and this class does not do them.
    - The reference drops one-point sequences when it selects the threshold. This class counts them, as the paper does. If two candidates give the same objective, the larger ``z`` wins, as in the reference.
    - The "learning from history" step of the paper needs labeled anomalies, and this class does not do it.

    Args:
        window_size (int): Number of recent smoothed scores, ``h`` in the paper (Default=2100).
        span (float): Span of the EWMA, so the smoothing factor is ``2 / (span + 1)`` (Default=105.0).
        p (float): Minimum percent decrease between the maxima of the anomalous sequences that pruning keeps (Default=0.13).
        z_values (tuple[float, ...]): Candidate numbers of standard deviations above the mean of the window (Default=(2.5, 3.0, ..., 10.0)).
    """

    def __init__(
        self,
        window_size: int = 2100,
        span: float = 105.0,
        p: float = 0.13,
        z_values: tuple[float, ...] = _Z_VALUES,
    ) -> None:
        self.window_size = window_size
        self.span = span
        self.p = p
        self.z_values = z_values
        self._z = np.sort(np.asarray(z_values, dtype=np.float64))
        self.smoothed_ = np.empty(window_size, dtype=np.float64)
        self.num_seen_ = 0
        self._ewma_sum = 0.0
        self._ewma_weight = 0.0

    def fit_partial(self, score: float) -> NonparametricDynamicThresholdPostprocessor:
        """Smooths the next score and adds it to the window.

        Args:
            score (float): Input score. It must be nonnegative.

        Returns:
            object: self.

        Raises:
            ValueError: If the score is negative.
        """
        if score < 0:
            raise ValueError(f"The scores must be nonnegative, got {score}.")
        decay = 1.0 - 2.0 / (self.span + 1.0)
        self._ewma_sum = score + decay * self._ewma_sum
        self._ewma_weight = 1.0 + decay * self._ewma_weight
        self.smoothed_[self.num_seen_ % self.window_size] = self._ewma_sum / self._ewma_weight
        self.num_seen_ += 1

        return self

    def transform_partial(self, score: float | None = None) -> float:
        """Gives the anomaly score of the last fitted score. This method should be used immediately after the fit_partial method with same score.

        Args:
            score (float): Ignored, the method uses the last fitted score.

        Returns:
            float: The score of the anomalous sequence that contains the last fitted score, or 0 if that score is nominal or no score was fitted.
        """
        if self.num_seen_ == 0:
            return 0.0
        if self.num_seen_ < self.window_size:
            window = self.smoothed_[: self.num_seen_]
        else:
            window = np.roll(self.smoothed_, -(self.num_seen_ % self.window_size))  # oldest first
        current = window[-1]
        mean = window.mean()
        std = window.std()
        thresholds = mean + self._z * std  # (num_z,)
        if not current > thresholds[0]:
            return 0.0

        thresholds = thresholds[thresholds < window.max()]  # candidates with anomalies
        above = window > thresholds[:, None]  # (num_z, window)
        num_anomalies = above.sum(axis=1)
        num_sequences = above[:, 0] + (above[:, 1:] & ~above[:, :-1]).sum(axis=1)
        # Mean and std of the scores below each threshold, centered at the window mean.
        below = np.where(above, 0.0, window - mean)  # (num_z, window)
        num_below = window.shape[0] - num_anomalies
        mean_shift = below.sum(axis=1) / num_below
        std_below = np.sqrt(np.maximum((below**2).sum(axis=1) / num_below - mean_shift**2, 0.0))
        objective = (-mean_shift / mean + (std - std_below) / std) / (
            num_anomalies + num_sequences**2
        )
        best = len(objective) - 1 - objective[::-1].argmax()  # largest z among ties
        anomalous = above[best]
        if not anomalous[-1]:
            return 0.0

        starts = np.flatnonzero(anomalous & ~np.r_[False, anomalous[:-1]])
        sequence_max = np.maximum.reduceat(np.where(anomalous, window, -np.inf), starts)
        e_max = np.append(np.sort(sequence_max)[::-1], window[~anomalous].max())
        decrease = (e_max[:-1] - e_max[1:]) / e_max[:-1]
        exceeded = np.flatnonzero(decrease > self.p)
        if exceeded.size == 0 or sequence_max[-1] < e_max[exceeded[-1]]:
            return 0.0

        return float((sequence_max[-1] - thresholds[best]) / (mean + std))
