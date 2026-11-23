from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pysad.core.base_metric import BaseMetric
from pysad.core.base_model import BaseModel
from pysad.core.base_postprocessor import BasePostprocessor


def _prequential_loop(
    scores: Iterable[float],
    y: Iterable[int],
    metrics: dict[str, BaseMetric],
    postprocessor: BasePostprocessor | None = None,
) -> dict[str, float | None]:
    """Transform each prequential score through the postprocessor and update every metric.

    Each score already comes from test-then-train scoring, so an instance is scored
    before anything learns from it. A postprocessor or calibrator learns from each score
    in the same pass, which keeps calibration prequential. Every metric records the label
    and the transformed score. This loop is shared by :func:`evaluate_stream` and
    :func:`evaluate_postprocessor`, so both stay consistent with a single implementation.

    Args:
        scores (Iterable[float]): One prequential score per instance, in stream order.
        y (Iterable[int]): One label per instance, in the same order. Use 1 for anomalies
            and 0 for normal instances.
        metrics (dict[str, BaseMetric]): The metrics to update, by name.
        postprocessor (BasePostprocessor or None): Optional calibrator or postprocessor
            applied to each score before the metrics record it (Default=None).

    Returns:
        dict[str, float | None]: The value of each metric at the end of the stream, by name.
    """
    for score, label in zip(scores, y, strict=True):
        if postprocessor is not None:
            score = postprocessor.fit_transform_partial(score)
        for metric in metrics.values():
            metric.update(label, score)
    return {name: metric.get() for name, metric in metrics.items()}


def evaluate_stream(
    model: BaseModel,
    X: Iterable[NDArray[Any]],
    y: Iterable[int],
    metrics: dict[str, BaseMetric],
    postprocessor: BasePostprocessor | None = None,
) -> dict[str, float | None]:
    """Evaluate a model on a labeled stream and return the metric values.

    For each instance, the model scores it before it learns from it. Then every metric
    records the label and score. When a postprocessor or calibrator is given, each score
    is transformed through it (and it learns from the score) before the metrics record it.
    The model never gets the labels.
    The function consumes both input iterables once, in stream order.
    It accepts arrays, lists, and single-use generators without loading the full stream into memory.
    Metrics can still keep all labels and scores.

    Args:
        model (BaseModel): The model to evaluate. It keeps the state that it learns from the stream.
        X (Iterable[np.ndarray]): Instances of shape (num_features,), in stream order.
        y (Iterable[int]): One label per instance, in the same order. Use 1 for anomalies and 0 for normal instances.
        metrics (dict[str, BaseMetric]): The metrics to update, by name.
        postprocessor (BasePostprocessor or None): Optional calibrator or postprocessor
            applied to each score before the metrics record it (Default=None).

    Returns:
        dict[str, float | None]: The value of each metric at the end of the stream, by name.

    Raises:
        ValueError: If the inputs have unequal lengths. The model and metrics keep updates from earlier pairs.
    """
    if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

    if postprocessor is None and isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        if X.ndim == 2 and X.dtype != object and y.ndim == 1:
            scores = model.fit_score(X)
            for metric in metrics.values():
                metric.update_batch(y, scores)
            return {name: metric.get() for name, metric in metrics.items()}

    return _prequential_loop(
        (model.fit_score_partial(x) for x in X),
        y,
        metrics,
        postprocessor,
    )


def evaluate_postprocessor(
    scores: Iterable[float],
    y: Iterable[int],
    metrics: dict[str, BaseMetric],
    postprocessor: BasePostprocessor,
) -> dict[str, float | None]:
    """Evaluate a postprocessor or calibrator on pre-computed scores, prequentially.

    Each score is transformed through the postprocessor (which learns from the score in
    the same pass) before every metric records it. This is postprocessor evaluation: it
    measures how well a calibrator or postprocessor transforms scores, for example with
    :class:`~pysad.evaluation.BrierScoreMetric`,
    :class:`~pysad.evaluation.ExpectedCalibrationErrorMetric`, or
    :class:`~pysad.evaluation.LogLossMetric`.

    The function consumes both input iterables once, in stream order. It accepts arrays,
    lists, and single-use generators without loading the full stream into memory.

    Args:
        scores (Iterable[float]): One precomputed score per instance, in stream order.
        y (Iterable[int]): One label per instance, in the same order. Use 1 for anomalies
            and 0 for normal instances.
        metrics (dict[str, BaseMetric]): The metrics to update, by name.
        postprocessor (BasePostprocessor): The calibrator or postprocessor to evaluate.

    Returns:
        dict[str, float | None]: The value of each metric at the end of the stream, by name.

    Raises:
        ValueError: If the inputs have unequal lengths. The postprocessor and metrics keep
            updates from earlier pairs.
    """
    if isinstance(scores, np.ndarray) and isinstance(y, np.ndarray):
        if len(scores) != len(y):
            raise ValueError("scores and y must have the same length.")

    return _prequential_loop(scores, y, metrics, postprocessor)
