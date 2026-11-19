from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.models.htm.encoders import (
    TIME_OF_DAY_SIZE,
    RandomDistributedScalarEncoder,
    encode_time_of_day,
)
from pysad.models.htm.spatial_pooler import SpatialPooler
from pysad.models.htm.temporal_memory import TemporalMemory
from pysad.transform.probability_calibration.anomaly_likelihood import (
    AnomalyLikelihoodProbabilityCalibrator,
)
from pysad.utils.validation import _finite_float, _int_at_least

# NAB's numenta detector pads the range of the values by this fraction on each side, and splits the
# padded range into this many buckets of the value encoder, each at least _MIN_RESOLUTION wide.
_RANGE_PADDING = 0.2
_NUM_BUCKETS = 130
_MIN_RESOLUTION = 0.001
# A value this fraction of the range of the earlier values outside that range is a spatial anomaly.
_SPATIAL_TOLERANCE = 0.05
_VALUE_SIZE = 400  # the bits of the value encoder


class HTM(BaseModel):
    """Scores each value of a timestamped univariate stream with Hierarchical Temporal Memory (HTM), the "numenta" detector of the Numenta Anomaly Benchmark (NAB) :cite:`ahmad2017unsupervised`.

    Each instance is the array ``[timestamp, value]``. The timestamp is in seconds since the Unix epoch, and the model uses only its time of day in UTC. To give a local time of day, pass the local time as if it were UTC, for example ``np.datetime64("2014-07-01T08:30").astype("datetime64[s]").astype(float)`` for a naive local time. ``Data().get_data(name, timestamps=True)`` gives the NAB series in this form.

    The model follows NAB's numenta detector, in four steps:

    1. An encoder gives the active bits of the time of day: 21 of 54 bits of a periodic scalar encoder with a radius of 9.49 hours. A random distributed scalar encoder gives 21 of 400 bits for the value. Its buckets are ``max(0.001, 1.4 * (max_val - min_val) / 130)`` wide, which is NAB's range padded by 20% on each side and split into 130 buckets. Close values share bits.
    2. The spatial pooler :cite:`cui2017htm` maps the bits to 40 active columns of ``num_columns``.
    3. The temporal memory :cite:`cui2016continuous` predicts the next active columns. The raw anomaly score is the fraction of the active columns that it did not predict (Eq. 1).
    4. :class:`~pysad.transform.probability_calibration.AnomalyLikelihoodProbabilityCalibrator` turns the raw scores into the anomaly likelihood. Its learning period is ``probationary_period // 2`` scores, and the rest of the probationary period gives its first estimate, as in NAB.

    A value that is more than 5% of the range of the earlier values outside that range is a spatial anomaly and gets the score 1.0, as in NAB. Higher scores are more anomalous. The paper reports an anomaly when the likelihood is at least ``1 - 1e-5``.

    The defaults are the parameters of NAB's numenta detector. The spatial pooler activates 40 columns, its columns have potential synapses to 80% of the bits, synapses are connected at a permanence of 0.2 and learn with steps of 0.003 and 0.0005, and it does not boost. The temporal memory has 32 cells per column, a segment is active with at least 13 connected synapses, a segment of a bursting column matches with at least 10 synapses to the previous active cells, and a segment has at most 32 synapses and grows up to 20 of them to the previous active cells. Synapses start at a permanence of 0.21, are connected above 0.5, and learn with steps of 0.1. A cell has at most 128 segments.

    Where the paper leaves details open or NAB differs from it, the model follows NAB:

    - The encoders and their parameters are those of NAB's numenta detector.
    - The spatial anomaly rule is not in the paper.
    - The calibrator follows NuPIC's anomaly likelihood. Its docstring lists how it differs from the paper.

    The implementation differs from NAB's numenta detector as follows:

    - NAB sets the range of the value encoder from the minimum and the maximum of the whole series. A stream gives them as ``min_val`` and ``max_val``.
    - The numenta detector uses NuPIC's backtracking temporal memory. This model uses the temporal memory of :cite:`cui2016continuous`, as NAB's numentaTM detector does, with the parameters of the numenta detector. The backtracking temporal memory does not punish the segments of wrong predictions, so the model does not either.
    - The calibrator does not use the values. NuPIC returns a likelihood near 0.5 when the variance of the values in its window is below ``1.5e-5``.
    - NAB returns the logarithm of the likelihood. The logarithm keeps the order of the scores, so a threshold on one gives the same alerts as the matching threshold on the other.
    - The random draws come from NumPy, so the encoder bits, the spatial pooler and the temporal memory differ from those of NuPIC with the same seed.

    On the 58 NAB series, with NAB's scoring code and seed 0, the model gets NAB scores of 64.2 (standard), 56.8 (low false alerts) and 68.3 (low misses). The NAB scoreboard lists 64.6, 56.7 and 69.2 for numentaTM, and 70.1, 63.1 and 74.3 for numenta, whose backtracking temporal memory this model does not port.

    Args:
        min_val (float): The smallest value that the stream is expected to have.
        max_val (float): The largest value that the stream is expected to have. A value outside the range gets its own bucket, up to 500 buckets on each side of the first value, and a value beyond them shares the last bucket.
        probationary_period (int): The number of first scores that the anomaly likelihood learns from before it leaves 0.5. NAB uses 15% of each series, at most 750 points (Default=750).
        likelihood (bool): Whether to return the anomaly likelihood. If False, the model returns the raw anomaly score, as NAB's numenta detector with ``useLikelihood = False`` (Default=True).
        num_columns (int): The number of columns of the spatial pooler and the temporal memory (Default=2048).
        num_cells_per_column (int): The number of cells in each column of the temporal memory (Default=32).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the encoder bits, the spatial pooler and the temporal memory. None draws from NumPy's global random state (Default=None).

    Raises:
        ValueError: If ``min_val`` is greater than ``max_val``.
    """

    _input_kind: ClassVar[Literal["multivariate", "univariate", "timestamped", "edge"]] = (
        "timestamped"
    )

    def __init__(
        self,
        min_val: float,
        max_val: float,
        probationary_period: int = 750,
        likelihood: bool = True,
        num_columns: int = 2048,
        num_cells_per_column: int = 32,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.min_val = _finite_float(min_val, "min_val")
        self.max_val = _finite_float(max_val, "max_val")
        if self.min_val > self.max_val:
            raise ValueError(f"min_val ({min_val}) must be at most max_val ({max_val}).")
        self.probationary_period = _int_at_least(probationary_period, "probationary_period", 1)
        self.likelihood = likelihood
        self.random_state = random_state

        span = (self.max_val - self.min_val) * (1.0 + 2.0 * _RANGE_PADDING)
        # NAB widens an empty range to 1.
        resolution = max(_MIN_RESOLUTION, (span if span > 0.0 else 1.0) / _NUM_BUCKETS)
        # One generator for the encoder, the spatial pooler and the temporal memory.
        rng = self._rng = check_random_state(random_state)
        self.encoder_ = RandomDistributedScalarEncoder(resolution, random_state=rng)
        self.spatial_pooler_ = SpatialPooler(
            num_inputs=TIME_OF_DAY_SIZE + _VALUE_SIZE, num_columns=num_columns, random_state=rng
        )
        self.temporal_memory_ = TemporalMemory(
            num_columns=num_columns,
            num_cells_per_column=num_cells_per_column,
            activation_threshold=12,  # NuPIC's "at least 13" is "more than 12" here
            learning_threshold=10,
            initial_permanence=0.21,
            connected_permanence=0.5,
            permanence_increment=0.1,
            permanence_decrement=0.1,
            predicted_segment_decrement=0.0,
            max_segments_per_cell=128,
            max_synapses_per_segment=32,
            max_new_synapses=20,
            random_state=rng,
        )
        learning_period = self.probationary_period // 2
        self.calibrator_ = AnomalyLikelihoodProbabilityCalibrator(
            learning_period=learning_period,
            num_estimation_samples=self.probationary_period - learning_period,
        )
        self.value_min_: float | None = None
        self.value_max_: float | None = None
        self._input = np.zeros(TIME_OF_DAY_SIZE + _VALUE_SIZE, dtype=bool)

    def __getstate__(self) -> dict:
        # NumPy's global random state is not restored from the pickle, so that the restored model
        # keeps drawing from it.
        state = self.__dict__.copy()
        state["_rng"] = None if self._rng is np.random.mtrand._rand else self._rng
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._rng = check_random_state(self._rng)
        self.encoder_._rng = self.temporal_memory_._rng = self._rng

    def _active_columns(self, timestamp: float, value: float, learn: bool) -> np.ndarray:
        self._input[:] = False
        self._input[encode_time_of_day(timestamp)] = True
        self._input[TIME_OF_DAY_SIZE + self.encoder_.encode(value, learn=learn)] = True
        return self.spatial_pooler_.compute(self._input, learn=learn)

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> HTM:
        """Learns the next instance.

        Args:
            X (np.float64 array of shape (2,)): The timestamp in seconds since the Unix epoch and the value.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        timestamp, value = X
        columns = self._active_columns(timestamp, value, learn=True)
        raw_score = self.temporal_memory_.score(columns)
        self.temporal_memory_.fit(columns)
        self.calibrator_.fit_partial(raw_score)
        if self.value_min_ is None or value < self.value_min_:
            self.value_min_ = float(value)
        if self.value_max_ is None or value > self.value_max_:
            self.value_max_ = float(value)
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the next instance with the predictions of the earlier instances.

        Args:
            X (np.float64 array of shape (2,)): The timestamp in seconds since the Unix epoch and the value.

        Returns:
            float: The anomaly likelihood of the instance, its raw anomaly score if ``likelihood`` is False, or 1.0 for a spatial anomaly.
        """
        timestamp, value = X
        if self.value_min_ is not None and self.value_max_ is not None:
            tolerance = (self.value_max_ - self.value_min_) * _SPATIAL_TOLERANCE
            if tolerance > 0.0 and not (
                self.value_min_ - tolerance <= value <= self.value_max_ + tolerance
            ):
                return 1.0
        raw_score = self.temporal_memory_.score(self._active_columns(timestamp, value, learn=False))
        return self.calibrator_.transform_partial(raw_score) if self.likelihood else raw_score

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Scores the next instance against earlier predictions and then learns it in one pass.

        Args:
            X (np.float64 array of shape (2,)): The timestamp in seconds since the Unix epoch and the value.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomaly likelihood of the instance, its raw anomaly score if ``likelihood`` is False, or 1.0 for a spatial anomaly.
        """
        if self.spatial_pooler_.boost_strength != 0.0:
            score = self.score_partial(X)
            self.fit_partial(X, y)
            return score

        timestamp, value = X
        spatial_anomaly = False
        if self.value_min_ is not None and self.value_max_ is not None:
            tolerance = (self.value_max_ - self.value_min_) * _SPATIAL_TOLERANCE
            spatial_anomaly = tolerance > 0.0 and not (
                self.value_min_ - tolerance <= value <= self.value_max_ + tolerance
            )

        columns = self._active_columns(timestamp, value, learn=True)
        raw_score = self.temporal_memory_.score(columns)
        if spatial_anomaly:
            score = 1.0
            self.calibrator_.fit_partial(raw_score)
        elif self.likelihood:
            score = self.calibrator_.fit_transform_partial(raw_score)
        else:
            score = raw_score
            self.calibrator_.fit_partial(raw_score)

        self.temporal_memory_.fit(columns)
        if self.value_min_ is None or value < self.value_min_:
            self.value_min_ = float(value)
        if self.value_max_ is None or value > self.value_max_:
            self.value_max_ = float(value)
        return float(score)
