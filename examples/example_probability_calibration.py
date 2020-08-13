from pysad.models import xStream
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
from pysad.utils import Data

# This example demonstrates the usage of the probability calibrators.
if __name__ == "__main__":
    model = xStream()
    calibrator = ConformalProbabilityCalibrator(windowed=True, window_size=300)
    streaming_data = Data().get_iterator("arrhythmia.mat")

    for i, (x, y_true) in enumerate(streaming_data):
        anomaly_score = model.fit_score_partial(x)

        calibrated_score = calibrator.fit_transform(anomaly_score)
        print(calibrated_score)
        if calibrated_score < 0.05: # ıf probabability is less than 5%.
            print(f"Alert: {i}th data point is anomalous.")
