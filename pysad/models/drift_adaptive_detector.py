from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from typing import Any, ClassVar

from numpy.typing import NDArray

from pysad.core.base_model import BaseModel
from pysad.statistics import PageHinkleyMeter


class DriftAdaptiveDetector(BaseModel):
    """Resets a detector when its scores trigger a drift alert.

    Uses the external change-monitoring approach of Bifet and Gavaldà
    :cite:`bifet2007learning`, with Page's cumulative statistic
    :cite:`page1954continuous`. This class composes existing detectors and meters.
    It does not define a new anomaly detection algorithm.

    Each update scores the instance, updates the meter, and then fits the instance.
    If the meter signals drift, the wrapper first creates a fresh detector and meter.
    The new detector learns the alerting instance once. The returned score comes
    from the old detector. ``score_partial`` changes neither detector nor meter.

    Differences from the papers: the wrapper monitors anomaly scores rather than
    classification errors. ``score_transform`` can map scores to bounded indicators.
    Page-Hinkley alerts when its two-sided statistic exceeds ``threshold``.
    Unlike the ADWIN paper's window-based retraining, resets retain no old instances.
    The meter also starts fresh because the new detector changes the score stream.
    Unlike River's default Page-Hinkley monitor, it adds no warmup or exponential decay.
    Non-finite transformed scores do not update the meter, but the detector still learns.

    Memory and update cost follow the detector. The Page-Hinkley monitor adds
    constant memory and update cost. The wrapper stores no stream history.
    Factories must return fresh detectors.
    For reproducible resets, use a factory with a fixed ``random_state``.
    For pickle support, use a top-level callable or ``functools.partial``.

    Args:
        model_factory (callable): A zero-argument callable that creates a fresh detector.
        drift_meter (PageHinkleyMeter or None): An unfitted Page-Hinkley meter to copy.
            None selects PageHinkleyMeter (Default=None).
        threshold (float): Positive Page-Hinkley alarm threshold (Default=50.0).
        score_transform (callable or None): Maps a score to the drift-monitoring value.
            None uses the score directly. Output scores always stay unchanged (Default=None).

    Raises:
        TypeError: If ``drift_meter`` is not a PageHinkleyMeter or None.

    Attributes:
        model_ (BaseModel): The current detector.
        drift_meter_ (PageHinkleyMeter): The current drift monitor.
        drift_detected_ (bool): Whether the last fit reset the detector.
        num_resets_ (int): The number of drift resets.
    """

    _is_detector: ClassVar[bool] = False

    def __init__(
        self,
        model_factory: Callable[[], BaseModel],
        drift_meter: PageHinkleyMeter | None = None,
        threshold: float = 50.0,
        score_transform: Callable[[float], float] | None = None,
    ) -> None:
        if drift_meter is not None and not isinstance(drift_meter, PageHinkleyMeter):
            raise TypeError("drift_meter must be a PageHinkleyMeter or None.")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive.")
        self.model_factory = model_factory
        self.drift_meter = deepcopy(drift_meter) if drift_meter is not None else PageHinkleyMeter()
        self.threshold = threshold
        self.score_transform = score_transform
        self.model_ = model_factory()
        self.drift_meter_ = deepcopy(self.drift_meter)
        self.drift_detected_ = False
        self.num_resets_ = 0

    def fit_partial(self, X: NDArray[Any], y: int | None = None) -> DriftAdaptiveDetector:
        """Updates the drift monitor and fits the instance after any reset.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to learn.
            y (int or None): The label to pass to the detector (Default=None).

        Returns:
            DriftAdaptiveDetector: The wrapper itself.
        """
        self.fit_score_partial(X, y)
        return self

    def score_partial(self, X: NDArray[Any]) -> float:
        """Returns the detector's score without learning or updating the meter.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: The anomaly score. Higher values mean more anomalous instances.
        """
        return self.model_.score_partial(X)

    def fit_score_partial(self, X: NDArray[Any], y: int | None = None) -> float:
        """Scores the instance before updating the meter and detector.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score and learn.
            y (int or None): The label to pass to the detector (Default=None).

        Returns:
            float: The score before learning or resetting.
        """
        score = self.model_.score_partial(X)
        value = score if self.score_transform is None else self.score_transform(score)
        self.drift_detected_ = False
        if math.isfinite(value):
            self.drift_meter_.update(value)
            self.drift_detected_ = self.drift_meter_.get() > self.threshold
        if self.drift_detected_:
            self.model_ = self.model_factory()
            self.drift_meter_ = deepcopy(self.drift_meter)
            self.num_resets_ += 1
        self.model_.fit_partial(X, y)
        return score
