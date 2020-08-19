# Import modules.
from pysad.models import xStream
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
from pysad.utils import Data
import numpy as np

# This example demonstrates the usage of the probability calibrators.
if __name__ == "__main__":
    np.random.seed(61)  # Fix seed.

    model = xStream()  # Init model.
    calibrator = ConformalProbabilityCalibrator(windowed=True, window_size=300)  # Init probability calibrator.
    streaming_data = Data().get_iterator("arrhythmia.mat")  # Get streamer.

    for i, (x, y_true) in enumerate(streaming_data):  # Stream data.
        anomaly_score = model.fit_score_partial(x)  # Fit to an instance x and score it.

        calibrated_score = calibrator.fit_transform(anomaly_score)  # Fit & calibrate score.

        # Output if the instance is anomalous.
        if calibrated_score > 0.95:  # If probability of being normal is less than 5%.
            print(f"Alert: {i}th data point is anomalous.")
