"""Flag anomalies in a SMAP telemetry channel with the nonparametric dynamic threshold of Telemanom."""

from pysad.evaluation import AUROCMetric
from pysad.models import MedianAbsoluteDeviation
from pysad.transform.postprocessing import NonparametricDynamicThresholdPostprocessor
from pysad.utils import Data

if __name__ == "__main__":
    model = MedianAbsoluteDeviation(window_size=250)
    postprocessor = NonparametricDynamicThresholdPostprocessor()
    metric = AUROCMetric()
    num_flagged = 0
    num_flagged_anomalies = 0

    for x, label in Data().get_iterator("smap/A-1.npy", shuffle=False):
        # The first feature is the telemetry value; the others are the commands.
        score = model.fit_score_partial(x[:1])
        # The output is 0 for a nominal point and the score of its anomalous sequence otherwise.
        anomaly_score = postprocessor.fit_transform_partial(score)
        metric.update(label, anomaly_score)
        num_flagged += anomaly_score > 0
        num_flagged_anomalies += (anomaly_score > 0) and label == 1

    print(f"Flagged points: {num_flagged}, of which in an anomaly range: {num_flagged_anomalies}")
    print(f"AUROC: {metric.get():.3f}")
