"""Score one-dimensional inputs before each update and calibrate the scores."""

import numpy as np

from pysad.models import HalfSpaceTrees
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
from pysad.utils import ArrayStreamer

if __name__ == "__main__":
    values = np.random.default_rng(0).normal(loc=20.0, scale=1.0, size=512)
    values[384] = 35.0
    X = values.reshape(-1, 1)  # One column holds one feature.

    model = HalfSpaceTrees(
        feature_mins=np.array([0.0]),
        feature_maxes=np.array([40.0]),
        window_size=64,
        num_trees=10,
        max_depth=8,
        random_state=0,
    )
    calibrator = ConformalProbabilityCalibrator(window_size=64)

    alerts = []
    for t, x in enumerate(ArrayStreamer().iter(X)):
        score = model.score_partial(x)
        calibrated = calibrator.fit_transform_partial(score)
        model.fit_partial(x)
        # Skip one model window and one calibration window.
        if t >= 128 and calibrated >= 0.95:
            alerts.append(t)

    print(f"{len(alerts)} alerts after warm-up")
