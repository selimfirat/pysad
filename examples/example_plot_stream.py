"""Plot the features, the calibrated scores with the threshold, and the alerts against the labels."""

import matplotlib.pyplot as plt
import numpy as np

from pysad.models import LODA
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
from pysad.utils import ArrayStreamer, generate_stream

if __name__ == "__main__":
    X, y = generate_stream(random_state=0)

    model = LODA(random_state=0)
    calibrator = ConformalProbabilityCalibrator(window_size=300)

    calibrated = np.array(
        [
            calibrator.fit_transform_partial(model.fit_score_partial(x))
            for x in ArrayStreamer().iter(X)
        ]
    )
    threshold = 0.99
    steps = np.arange(len(calibrated))
    alerts = steps[calibrated > threshold]

    fig, (features_ax, scores_ax, alerts_ax) = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    features_ax.plot(steps, X, linewidth=0.5)
    features_ax.set_ylabel("features")
    scores_ax.plot(steps, calibrated, linewidth=0.8, label="calibrated score")
    scores_ax.axhline(threshold, color="tab:red", linestyle="--", label="threshold")
    scores_ax.set_ylabel("score")
    scores_ax.legend()
    alerts_ax.eventplot(
        [alerts, steps[y == 1]],
        lineoffsets=[0, 1],
        linelengths=0.8,
        colors=["tab:orange", "tab:red"],
    )
    alerts_ax.set_yticks([0, 1], ["alert", "anomaly"])
    alerts_ax.set_xlabel("position in the stream")
    fig.tight_layout()
    plt.show()
