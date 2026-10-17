"""Preprocess each point, score it with xStream and smooth the scores."""

from pysad.evaluation import AUROCMetric
from pysad.models import xStream
from pysad.transform.postprocessing import RunningAveragePostprocessor
from pysad.transform.preprocessing import InstanceUnitNormScaler
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    scaler = InstanceUnitNormScaler()
    model = xStream(random_state=61)
    smoother = RunningAveragePostprocessor(window_size=5)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        scaled = scaler.fit_transform_partial(x)
        score = model.fit_score_partial(scaled)
        smoothed = smoother.fit_transform_partial(score)
        metric.update(label, smoothed)

    print(f"AUROC: {metric.get():.3f}")
