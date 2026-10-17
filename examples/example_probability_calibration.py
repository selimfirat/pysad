"""Convert scores into values between 0 and 1 and raise alerts above a threshold."""

from pysad.models import LODA
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    model = LODA(random_state=61)
    calibrator = ConformalProbabilityCalibrator(window_size=300)

    alerts = []
    for t, x in enumerate(ArrayStreamer().iter(X)):
        score = model.fit_score_partial(x)
        calibrated = calibrator.fit_transform_partial(score)
        # 0.95 means that the score is higher than 95% of the last 300 scores.
        # Skip the first 100 points, while the model and the calibrator warm up.
        if t >= 100 and calibrated > 0.95:
            alerts.append(t)

    print(f"{len(alerts)} alerts at points {alerts}")
