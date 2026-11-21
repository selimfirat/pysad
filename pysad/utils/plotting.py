"""Streaming plot helpers for features, scores and alerts.

The helpers import matplotlib only when they run, so ``pysad`` stays importable
without it. The helpers draw a figure that a caller can show or save.
"""

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

_FEATURE_COLORS = ("#00757a", "#2a9d9f", "#5c6bc0", "#64748b", "#c26d1f", "#00565a")
_SCORE_COLOR = "#00757a"
_BAND_COLOR = "#99d5d2"
_ALERT_COLOR = "#e63946"
_ANOMALY_COLOR = "#c26d1f"
_GRID_COLOR = "#dce3e2"


class _HasThreshold(Protocol):
    """Protocol for calibrators that expose a threshold_ property."""

    def fit_partial(self, score: float) -> Any: ...

    @property
    def threshold_(self) -> float: ...


def plot_stream(
    X: NDArray[np.float64],
    scores: NDArray[np.float64],
    *,
    threshold: float | None = None,
    lower: NDArray[np.float64] | None = None,
    upper: NDArray[np.float64] | None = None,
    alerts: NDArray[np.bool_] | None = None,
    labels: NDArray[np.int64] | None = None,
    figsize: tuple[float, float] = (10.0, 7.0),
) -> tuple[Any, tuple[Any, ...]]:
    """Plots the stream features, the scores with a confidence band, and the alerts.

    The helper draws three panels that share the stream position on the x-axis:
    the features, the scores with an optional confidence band and threshold, and
    the alert spikes next to the true anomaly labels.

    Args:
        X (NDArray[np.float64]): The features of the stream, shape (num_instances, num_features).
        scores (NDArray[np.float64]): The anomaly scores, shape (num_instances,).
        threshold (float | None): The alert threshold. The helper draws it as a
            horizontal line (Default=None).
        lower (NDArray[np.float64] | None): The lower bound of the confidence band,
            shape (num_instances,) (Default=None).
        upper (NDArray[np.float64] | None): The upper bound of the confidence band,
            shape (num_instances,) (Default=None).
        alerts (NDArray[np.bool_] | None): The stream positions that raised an alert (Default=None).
        labels (NDArray[np.int64] | None): The true anomaly labels, shape (num_instances,)
            (Default=None).
        figsize (tuple[float, float]): The size of the figure (Default=(10.0, 7.0)).

    Returns:
        tuple[matplotlib.figure.Figure, tuple[matplotlib.axes.Axes, ...]]: The
            figure and its three axes: features, scores and alerts.
    """
    import matplotlib.pyplot as plt

    steps = np.arange(len(scores))

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, layout="constrained")
    features_ax, scores_ax, alerts_ax = axes

    features_ax.set_prop_cycle(color=list(_FEATURE_COLORS))
    features_ax.plot(steps, X, linewidth=0.5)
    features_ax.set_ylabel("features")
    features_ax.yaxis.grid(True, color=_GRID_COLOR, linewidth=0.6, alpha=0.8)

    scores_ax.plot(steps, scores, color=_SCORE_COLOR, linewidth=0.8, label="score")
    if lower is not None and upper is not None:
        scores_ax.fill_between(
            steps,
            lower,
            upper,
            color=_BAND_COLOR,
            alpha=0.35,
            label="confidence band",
        )
    if threshold is not None:
        scores_ax.axhline(threshold, color=_ALERT_COLOR, linestyle="--", label="threshold")
    scores_ax.set_ylabel("score")
    scores_ax.yaxis.grid(True, color=_GRID_COLOR, linewidth=0.6, alpha=0.8)
    scores_ax.legend()

    positions: list[NDArray[np.int64]] = []
    offsets: list[int] = []
    names: list[str] = []
    colors: list[str] = []
    if alerts is not None:
        alert_positions = steps[np.asarray(alerts)]
        if len(alert_positions) > 0:
            positions.append(alert_positions)
            offsets.append(len(offsets))
            names.append("alert")
            colors.append(_ALERT_COLOR)
    if labels is not None:
        anomaly_positions = steps[np.asarray(labels) == 1]
        if len(anomaly_positions) > 0:
            positions.append(anomaly_positions)
            offsets.append(len(offsets))
            names.append("anomaly")
            colors.append(_ANOMALY_COLOR)
    if positions:
        alerts_ax.eventplot(
            positions,
            lineoffsets=offsets,
            linelengths=0.8,
            colors=colors,
        )
        alerts_ax.set_yticks(offsets, names)
    alerts_ax.set_xlabel("position in the stream")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return fig, axes


def plot_streaming_ribbon(
    X: NDArray[np.float64],
    scores: NDArray[np.float64],
    calibrator: _HasThreshold,
    *,
    labels: NDArray[np.int64] | None = None,
    figsize: tuple[float, float] = (10.0, 7.0),
    ribbon_lower: float | NDArray[np.float64] | None = 0.0,
) -> tuple[Any, tuple[Any, ...]]:
    """Plots the stream features, the raw scores with a conformal ribbon, and the alerts.

    The helper draws three panels that share the stream position on the x-axis:
    the features, the raw anomaly scores with a conformal prediction ribbon, and
    the alert spikes next to the true anomaly labels.

    The conformal ribbon shows the adaptive prediction interval from the calibrator.
    For each time step, the ribbon spans from ``ribbon_lower`` to the calibrator's
    ``threshold_`` (the upper conformal quantile). Scores above the ribbon trigger
    alerts.

    The calibrator is deep-copied and run on the scores in stream order to obtain
    the threshold at each step. The original calibrator is not modified.

    Args:
        X (NDArray[np.float64]): The features of the stream, shape (num_instances, num_features).
        scores (NDArray[np.float64]): The raw anomaly scores, shape (num_instances,).
        calibrator: A conformal calibrator instance with a ``threshold_`` property
            and a ``fit_partial`` method (e.g. ``ConformalTSFMCalibrator``).
        labels (NDArray[np.int64] | None): The true anomaly labels, shape (num_instances,)
            (Default=None).
        figsize (tuple[float, float]): The size of the figure (Default=(10.0, 7.0)).
        ribbon_lower (float | NDArray[np.float64] | None): The lower bound of the ribbon.
            Can be a scalar (applied to all steps) or an array of shape
            (num_instances,). Use ``None`` to omit the lower bound and draw only
            the upper threshold line (Default=0.0).

    Returns:
        tuple[matplotlib.figure.Figure, tuple[matplotlib.axes.Axes, ...]]: The
            figure and its three axes: features, scores and alerts.
    """
    import copy

    import matplotlib.pyplot as plt

    steps = np.arange(len(scores))

    # Deep copy the calibrator and run it on the scores to get thresholds.
    cal = copy.deepcopy(calibrator)
    thresholds = np.empty(len(scores), dtype=np.float64)
    for i, s in enumerate(scores):
        thresholds[i] = cal.threshold_
        cal.fit_partial(float(s))

    # Determine alerts where scores exceed the threshold.
    alerts = np.where(scores > thresholds)[0]

    # Prepare lower bound for the ribbon.
    lower: NDArray[np.float64] | None
    if ribbon_lower is None:
        lower = None
    elif isinstance(ribbon_lower, (int, float, np.number)):
        lower = np.full(len(scores), float(ribbon_lower))
    else:
        lower = np.asarray(ribbon_lower, dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, layout="constrained")
    features_ax, scores_ax, alerts_ax = axes

    features_ax.set_prop_cycle(color=list(_FEATURE_COLORS))
    features_ax.plot(steps, X, linewidth=0.5)
    features_ax.set_ylabel("features")
    features_ax.yaxis.grid(True, color=_GRID_COLOR, linewidth=0.6, alpha=0.8)

    scores_ax.plot(steps, scores, color=_SCORE_COLOR, linewidth=0.8, label="score")
    if lower is not None:
        scores_ax.fill_between(
            steps,
            lower,
            thresholds,
            color=_BAND_COLOR,
            alpha=0.35,
            label="conformal ribbon",
        )
    else:
        scores_ax.plot(steps, thresholds, color=_ALERT_COLOR, linestyle="--", label="threshold")
    scores_ax.set_ylabel("score")
    scores_ax.yaxis.grid(True, color=_GRID_COLOR, linewidth=0.6, alpha=0.8)
    scores_ax.legend()

    positions: list[NDArray[np.int64]] = []
    offsets: list[int] = []
    names: list[str] = []
    colors: list[str] = []
    if len(alerts) > 0:
        positions.append(alerts)
        offsets.append(len(offsets))
        names.append("alert")
        colors.append(_ALERT_COLOR)
    if labels is not None:
        anomaly_positions = steps[np.asarray(labels) == 1]
        if len(anomaly_positions) > 0:
            positions.append(anomaly_positions)
            offsets.append(len(offsets))
            names.append("anomaly")
            colors.append(_ANOMALY_COLOR)
    if positions:
        alerts_ax.eventplot(
            positions,
            lineoffsets=offsets,
            linelengths=0.8,
            colors=colors,
        )
        alerts_ax.set_yticks(offsets, names)
    alerts_ax.set_xlabel("position in the stream")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return fig, axes
