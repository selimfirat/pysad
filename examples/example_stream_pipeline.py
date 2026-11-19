"""Chain a scaler, xStream, a running z-score and a calibrator into one model."""

from pysad.evaluation import AUROCMetric
from pysad.models import StreamPipeline, xStream
from pysad.transform.postprocessing import ZScorePostprocessor
from pysad.transform.preprocessing import InstanceUnitNormScaler
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    pipeline = StreamPipeline(
        model=xStream(random_state=0),
        preprocessors=[InstanceUnitNormScaler()],
        postprocessors=[
            ZScorePostprocessor(window_size=100),
            ConformalProbabilityCalibrator(window_size=300),
        ],
    )
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        # A value between 0 and 1: the share of the last 300 normalized scores below this one.
        calibrated = pipeline.fit_score_partial(x)
        metric.update(label, calibrated)

    print(f"AUROC: {metric.get():.3f}")
