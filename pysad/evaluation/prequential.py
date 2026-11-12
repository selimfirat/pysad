from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from pysad.core.base_metric import BaseMetric
from pysad.core.base_model import BaseModel


def evaluate_stream(
    model: BaseModel,
    X: Iterable[np.ndarray],
    y: Iterable[int],
    metrics: dict[str, BaseMetric],
) -> dict[str, float | None]:
    """Evaluate a model on a labeled stream and return the metric values.

    For each instance, the model scores it before it learns from it.
    Then every metric records the label and score. The model never gets the labels.
    The function consumes both input iterables once, in stream order.
    It accepts arrays, lists, and single-use generators without loading the full stream into memory.
    Metrics can still keep all labels and scores.

    Args:
        model (BaseModel): The model to evaluate. It keeps the state that it learns from the stream.
        X (Iterable[np.ndarray]): Instances of shape (num_features,), in stream order.
        y (Iterable[int]): One label per instance, in the same order. Use 1 for anomalies and 0 for normal instances.
        metrics (dict[str, BaseMetric]): The metrics to update, by name.

    Returns:
        dict[str, float | None]: The value of each metric at the end of the stream, by name.

    Raises:
        ValueError: If the inputs have unequal lengths. The model and metrics keep updates from earlier pairs.
    """
    if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if X.ndim == 2 and X.dtype != object and y.ndim == 1:
            scores = model.fit_score(X)
            for metric in metrics.values():
                metric.update_batch(y, scores)
            return {name: metric.get() for name, metric in metrics.items()}

    for x, label in zip(X, y, strict=True):
        score = model.fit_score_partial(x)
        for metric in metrics.values():
            metric.update(label, score)

    return {name: metric.get() for name, metric in metrics.items()}
