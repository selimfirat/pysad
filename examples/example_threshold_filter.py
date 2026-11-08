"""Turn calibrated scores into 0/1 labels with a fixed threshold."""

from pysad.evaluation import PrecisionMetric, RecallMetric
from pysad.models import LODA
from pysad.transform.postprocessing import ThresholdFilter
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
from pysad.utils import Data

if __name__ == "__main__":
    model = LODA(random_state=0)
    calibrator = ConformalProbabilityCalibrator(window_size=300)
    threshold_filter = ThresholdFilter(threshold=0.95)
    precision = PrecisionMetric()
    recall = RecallMetric()

    for x, label in Data().get_iterator("arrhythmia.mat", shuffle=False):
        calibrated = calibrator.fit_transform_partial(model.fit_score_partial(x))
        # The label is 1 if the score is higher than 95% of the last 300 scores.
        predicted = threshold_filter.fit_transform_partial(calibrated)
        precision.update(label, predicted)
        recall.update(label, predicted)

    print(f"Precision: {precision.get():.3f}, recall: {recall.get():.3f}")
