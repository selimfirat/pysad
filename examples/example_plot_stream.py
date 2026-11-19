"""Plot the features, the calibrated scores with the threshold, and the alerts against the labels."""

import matplotlib.pyplot as plt
import numpy as np

from pysad.models import LODA
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
from pysad.utils import ArrayStreamer, generate_stream, plot_stream

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
    alerts = np.arange(len(calibrated))[calibrated > threshold]

    fig, _ = plot_stream(
        X,
        calibrated,
        threshold=threshold,
        alerts=alerts,
        labels=y,
    )
    plt.show()
