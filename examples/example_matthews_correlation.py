"""Measure binary predictions when anomalies are rare."""

from pysad.evaluation import MatthewsCorrelationMetric
from pysad.models import LODA
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("cardio.mat")

    model = LODA(random_state=0)
    calibrator = ConformalProbabilityCalibrator(window_size=1000)
    metric = MatthewsCorrelationMetric(threshold=0.95)

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        calibrated = calibrator.fit_transform_partial(score)
        metric.update(label, calibrated)

    print(f"Matthews correlation: {metric.get():.3f}")
