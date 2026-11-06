from __future__ import annotations

import numpy as np

from pysad.core.base_metric import BaseMetric
from pysad.core.base_model import BaseModel


def evaluate_stream(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    metrics: dict[str, BaseMetric],
) -> dict[str, float | None]:
    """Runs prequential evaluation of a model on a labeled stream and returns the metric values.

    For each instance in order, the model scores the instance, every metric records the label and the score, and then the model learns from the instance. The model never gets the labels. This is the test-then-train order of ``progressive_val_score`` in River.

    Args:
        model (BaseModel): The model to evaluate. It keeps the state that it learned from the stream.
        X (np.array of shape (num_instances, num_features)): The instances, in stream order.
        y (np.array of shape (num_instances,)): The labels, 1 for an anomaly and 0 for a normal instance.
        metrics (dict[str, BaseMetric]): The metrics to update, by name.

    Returns:
        dict[str, float | None]: The value of each metric at the end of the stream, by name.
    """
    for x, label in zip(X, y, strict=True):
        score = model.fit_score_partial(x)
        for metric in metrics.values():
            metric.update(label, score)

    return {name: metric.get() for name, metric in metrics.items()}
