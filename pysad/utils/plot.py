from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_stream(
    X: np.ndarray,
    scores: np.ndarray,
    threshold: float | np.ndarray,
    y: np.ndarray | None = None,
) -> Figure:
    """Plots a scored stream in one figure with three panels that share the stream position.

    The top panel shows each feature, the middle panel shows the scores and the threshold, and the bottom panel shows the alerts, the positions where the score is above the threshold, against the anomaly labels.

    The scores can be calibrated values, for example from :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator` with a fixed threshold such as 0.95, or raw scores with a threshold that changes over the stream, for example ``threshold_`` of :class:`~pysad.transform.probability_calibration.SPOTProbabilityCalibrator` after each instance.

    This function needs matplotlib, which PySAD does not install. Install it with ``pip install matplotlib``.

    Args:
        X (np.float64 array of shape (num_instances, num_features)): The features in stream order.
        scores (np.float64 array of shape (num_instances,)): The score of each instance.
        threshold (float or np.float64 array of shape (num_instances,)): The alert threshold, one value for the whole stream or one value for each instance.
        y (np.int32 array of shape (num_instances,)): The labels, 1 for an anomaly and 0 otherwise. If None, the bottom panel shows only the alerts (Default=None).

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    import matplotlib.pyplot as plt  # matplotlib is not a PySAD dependency, so import it only here.

    steps = np.arange(len(scores))
    thresholds = np.broadcast_to(threshold, steps.shape)
    alerts = steps[scores > thresholds]

    fig, (features_ax, scores_ax, alerts_ax) = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    features_ax.plot(steps, X, linewidth=0.5)
    features_ax.set_ylabel("features")

    scores_ax.plot(steps, scores, linewidth=0.8, label="score")
    scores_ax.plot(steps, thresholds, color="tab:red", linestyle="--", label="threshold")
    scores_ax.set_ylabel("score")
    scores_ax.legend()

    positions, names, colors = [alerts], ["alert"], ["tab:orange"]
    if y is not None:
        positions.append(steps[y == 1])
        names.append("anomaly")
        colors.append("tab:red")
    alerts_ax.eventplot(positions, lineoffsets=range(len(names)), linelengths=0.8, colors=colors)
    alerts_ax.set_yticks(range(len(names)), names)
    alerts_ax.set_xlabel("position in the stream")

    fig.tight_layout()
    return fig
