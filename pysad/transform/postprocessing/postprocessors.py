from __future__ import annotations

from typing import cast

import numpy as np

from pysad.core.base_postprocessor import BasePostprocessor
from pysad.core.base_statistic import BaseStatistic
from pysad.statistics.basic import AverageMeter, MaxMeter, VarianceMeter
from pysad.statistics.quantiles import MedianMeter, QuantileMeter
from pysad.statistics.running_statistic import RunningStatistic
from pysad.utils.window import RingBuffer

_Z_VALUES = tuple(z / 2 for z in range(5, 21))


class _MeterPostprocessor(BasePostprocessor):
    """Base class for postprocessors that convert a score to a statistic of the previous scores.

    Args:
        meter: The statistic to update with each score.
    """

    def __init__(self, meter: BaseStatistic) -> None:
        self.meter = meter
        self.num_seen_ = 0

    def fit_partial(self, score: float) -> _MeterPostprocessor:
        """Fits the postprocessor to the (next) timestep's score.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.meter.update(score)
        self.num_seen_ += 1

        return self

    def transform_partial(self, score: float | None = None) -> float:
        """Applies postprocessing to the score.

        Args:
            score (float): The input score.

        Returns:
            float: Transformed score, or 0.0 before the first fit.
        """
        if self.num_seen_ == 0:
            return 0.0
        return self.meter.get()

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Transforms a batch of scores using the current frozen meter statistic."""
        n = len(scores)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        val = 0.0 if self.num_seen_ == 0 else float(self.meter.get())
        return np.full(n, val, dtype=np.float64)


class AveragePostprocessor(_MeterPostprocessor):
    """Converts a score to the average of previous scores over a sliding window or the full stream.

    Args:
        window_size (int or None): Length of the sliding window of recent scores. If None, uses all previous scores (Default=None).
    """

    def __init__(self, window_size: int | None = None) -> None:
        self.window_size = window_size
        meter: BaseStatistic = (
            AverageMeter()
            if window_size is None
            else RunningStatistic(statistic_cls=AverageMeter, window_size=window_size)
        )
        super().__init__(meter)

    def fit(self, scores: np.ndarray) -> AveragePostprocessor:
        if (
            self.window_size is None
            and isinstance(scores, np.ndarray)
            and scores.ndim == 1
            and not np.isnan(scores).any()
        ):
            if len(scores) > 0:
                meter = cast(AverageMeter, self.meter)
                meter.sum += float(scores.astype(np.float64, copy=False).sum())
                meter.num_items += len(scores)
                self.num_seen_ += len(scores)
            return self
        return cast(AveragePostprocessor, super().fit(scores))

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        if (
            self.window_size is None
            and isinstance(scores, np.ndarray)
            and scores.ndim == 1
            and not np.isnan(scores).any()
        ):
            if len(scores) == 0:
                return np.empty(0, dtype=np.float64)
            arr = scores.astype(np.float64, copy=False)
            meter = cast(AverageMeter, self.meter)
            cumsum = meter.sum + arr.cumsum()
            counts = np.arange(
                meter.num_items + 1, meter.num_items + len(arr) + 1, dtype=np.float64
            )
            meter.sum = float(cumsum[-1])
            meter.num_items += len(arr)
            self.num_seen_ += len(arr)
            return cumsum / counts
        return super().fit_transform(scores)


class MaxPostprocessor(_MeterPostprocessor):
    """Converts a score to the maximum of previous scores over a sliding window or the full stream.

    Args:
        window_size (int or None): Length of the sliding window of recent scores. If None, uses all previous scores (Default=None).
    """

    def __init__(self, window_size: int | None = None) -> None:
        self.window_size = window_size
        meter: BaseStatistic = (
            MaxMeter()
            if window_size is None
            else RunningStatistic(statistic_cls=MaxMeter, window_size=window_size)
        )
        super().__init__(meter)

    def fit(self, scores: np.ndarray) -> MaxPostprocessor:
        if (
            self.window_size is None
            and isinstance(scores, np.ndarray)
            and scores.ndim == 1
            and not np.isnan(scores).any()
        ):
            if len(scores) > 0:
                meter = cast(MaxMeter, self.meter)
                batch_max = float(scores.max())
                if batch_max > meter.max:
                    meter.max = batch_max
                self.num_seen_ += len(scores)
            return self
        return cast(MaxPostprocessor, super().fit(scores))

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        if (
            self.window_size is None
            and isinstance(scores, np.ndarray)
            and scores.ndim == 1
            and not np.isnan(scores).any()
        ):
            if len(scores) == 0:
                return np.empty(0, dtype=np.float64)
            arr = scores.astype(np.float64, copy=False)
            meter = cast(MaxMeter, self.meter)
            acc = np.maximum(meter.max, np.maximum.accumulate(arr))
            meter.max = float(acc[-1])
            self.num_seen_ += len(arr)
            return acc
        return super().fit_transform(scores)


class MedianPostprocessor(_MeterPostprocessor):
    """Converts a score to the median of previous scores over a sliding window or the full stream.

    Args:
        window_size (int or None): Length of the sliding window of recent scores. If None, computes the exact median over all previous scores by retaining them; pass ``window_size`` to bound memory on long streams (Default=None).
    """

    def __init__(self, window_size: int | None = None) -> None:
        self.window_size = window_size
        meter: BaseStatistic = (
            MedianMeter()
            if window_size is None
            else RunningStatistic(statistic_cls=MedianMeter, window_size=window_size)
        )
        super().__init__(meter)


class ZScorePostprocessor(BasePostprocessor):
    """Normalizes scores via Z-score normalization over a sliding window or the full stream.

    Args:
        window_size (int or None): Length of the sliding window of recent scores. If None, uses all previous scores (Default=None).
    """

    def __init__(self, window_size: int | None = None) -> None:
        self.window_size = window_size
        if window_size is None:
            variance_meter = VarianceMeter()
            self.meter: BaseStatistic = variance_meter
            self._variance_meter = variance_meter
        else:
            running = RunningStatistic(statistic_cls=VarianceMeter, window_size=window_size)
            self.meter = running
            self._variance_meter = cast(VarianceMeter, running.statistic)
        self.num_seen_ = 0

    def fit_partial(self, score: float) -> ZScorePostprocessor:
        """Fits the postprocessor to the (next) timestep's score.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.meter.update(score)
        self.num_seen_ += 1

        return self

    def transform_partial(self, score: float) -> float:
        """Normalizes ``score`` using the tracked mean and standard deviation.

        Args:
            score (float): The input score.

        Returns:
            float: Normalized score, or 0.0 before the first fit or when variance is zero.
        """
        if self.num_seen_ == 0:
            return 0.0
        variance = self._variance_meter.get()
        # Scores in a zero-variance window equal the mean, so their normalized
        # deviation is zero. Returning 0.0 also keeps downstream ensemblers usable.
        if variance == 0:
            return 0.0

        return float((score - self._variance_meter.mean) / np.sqrt(variance))

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Normalizes a batch of scores using the current frozen mean and variance."""
        if isinstance(scores, np.ndarray) and scores.ndim == 1:
            n = len(scores)
            if n == 0 or self.num_seen_ == 0:
                return np.zeros(n, dtype=np.float64)
            variance = self._variance_meter.get()
            if variance == 0:
                return np.zeros(n, dtype=np.float64)
            mean = self._variance_meter.mean
            return np.asarray((scores - mean) / np.sqrt(variance), dtype=np.float64)
        return super().transform(scores)


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

    Raises:
        ValueError: If ``window_size`` is less than 1.
    """

    def __init__(
        self,
        window_size: int = 2100,
        span: float = 105.0,
        p: float = 0.13,
        z_values: tuple[float, ...] = _Z_VALUES,
    ) -> None:
        self._buffer = RingBuffer(window_size)
        self.window_size = window_size
        self.span = span
        self.p = p
        self.z_values = z_values
        self._z = np.sort(np.asarray(z_values, dtype=np.float64))
        self.num_seen_ = 0
        self._ewma_sum = 0.0
        self._ewma_weight = 0.0
        self._current = 0.0

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
        self._current = self._ewma_sum / self._ewma_weight
        self._buffer.append(self._current)
        self.num_seen_ += 1

        return self

    def transform_partial(self, score: float | None = None) -> float:
        """Gives the anomaly score of the last fitted score. This method should be used immediately after the fit_partial method with same score.

        Args:
            score (float): Ignored, the method uses the last fitted score.

        Returns:
            float: The score of the anomalous sequence that contains the last fitted score.
                Returns 0 if that score is nominal or before the first fit.
        """
        if self.num_seen_ == 0:
            return 0.0
        raw = self._buffer.points[: self._buffer.count]
        approx_thresh = raw.mean() + self._z[0] * raw.std()
        if not self._current > approx_thresh - 1e-12 * abs(approx_thresh) - 1e-15:
            return 0.0

        window = self._buffer.to_array()
        mean = window.mean()
        std = window.std()
        thresholds = mean + self._z * std
        if not window[-1] > thresholds[0]:
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

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Transforms a batch using the frozen window's last-score anomaly value."""
        n = len(scores)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        return np.full(n, self.transform_partial(), dtype=np.float64)


class QuantileFilter(BasePostprocessor):
    """A postprocessor that labels a score 1 if it is at least the ``q`` quantile of the previous scores, and 0 otherwise.

    This is the ``QuantileFilter`` of River (https://github.com/online-ml/river, BSD-3-Clause license). The quantile comes from :class:`~pysad.statistics.QuantileMeter`, the P^2 algorithm of :cite:`jain1985psquare`, so the memory stays constant.
    ``fit_transform_partial`` labels the score with the quantile of the scores before it, then fits the score, as River does.

    These are the differences from River:

    - River labels a score 0 when the quantile of the previous scores is exactly 0. This class labels it 1 if the score is at least 0.
    - River does not learn the detector from the scores that it labels 1, unless ``protect_anomaly_detector=False``. A postprocessor does not own the detector, so this class leaves that choice to the caller, see :doc:`/guides/alerts`.
    - With 2 to 5 previous scores, the quantile can differ from River's. :class:`~pysad.statistics.QuantileMeter` interpolates between the sorted scores for the first four scores and uses the central P^2 marker from the fifth score, as the paper does. River's ``Quantile`` takes one of the sorted scores until the sixth score. With more previous scores, the two quantiles and the labels are the same.

    Args:
        q (float): The quantile of the previous scores that a score must reach to get the label 1, in the open interval (0, 1) (Default=0.95).
    """

    def __init__(self, q: float = 0.95) -> None:
        self.meter = QuantileMeter(q=q)

    def fit_partial(self, score: float) -> QuantileFilter:
        """Updates the quantile with the next score.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.meter.update(score)

        return self

    def transform_partial(self, score: float) -> float:
        """Labels the score with the quantile of the fitted scores.

        Args:
            score (float): The input score.

        Returns:
            float: 1.0 if the score is at least the quantile of the fitted scores, and 0.0 otherwise or if no score was fitted.
        """
        return float(self.meter.num_items > 0 and score >= self.meter.get())

    def fit_transform_partial(self, score: float) -> float:
        """Labels the score with the quantile of the previous scores, then fits the score.

        Args:
            score (float): Input score.

        Returns:
            float: 1.0 if the score is at least the quantile of the previous scores, and 0.0 otherwise.
        """
        label = self.transform_partial(score)
        self.fit_partial(score)
        return label

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Labels a batch of scores against the current frozen quantile."""
        arr = np.asarray(scores)
        if arr.ndim == 1:
            if len(arr) == 0 or self.meter.num_items == 0:
                return np.zeros(len(arr), dtype=np.float64)
            q_val = self.meter.get()
            return (arr >= q_val).astype(np.float64)
        return super().transform(scores)


class ThresholdFilter(BasePostprocessor):
    """A postprocessor that turns a score into a label with a fixed threshold, like ``ThresholdFilter`` of River.

    The output is 1.0 (anomalous) if the score is at least ``threshold``, and 0.0 (normal) otherwise. Unlike River, the filter does not wrap a detector and does not keep the anomalies out of the detector. To keep them out, call ``fit_partial`` of the model only for the points with the output 0.

    Args:
        threshold (float): The lowest score that gets the label 1.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def fit_partial(self, score: float) -> ThresholdFilter:
        """Does nothing, because the threshold is fixed.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        return self

    def transform_partial(self, score: float) -> float:
        """Labels the score.

        Args:
            score (float): The input score.

        Returns:
            float: 1.0 if the score is at least ``threshold``, 0.0 otherwise.
        """
        return float(score >= self.threshold)

    def fit(self, scores: np.ndarray) -> ThresholdFilter:
        """Does nothing, because the threshold is fixed."""
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Labels all scores in a 1-D batch at once."""
        arr = np.asarray(scores)
        if arr.ndim == 1:
            return (arr >= self.threshold).astype(np.float64)
        return super().transform(scores)

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        """Labels all scores in a 1-D batch at once."""
        return self.transform(scores)
