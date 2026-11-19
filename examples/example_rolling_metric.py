"""Evaluate a detector on the most recent labeled instances."""

from pysad.evaluation import RecallMetric, RollingMetric
from pysad.models import xStream
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")
    model = xStream(random_state=0)
    metric = RollingMetric(metric_cls=RecallMetric, window_size=100, threshold=0.5)

    print(f"Empty window: {metric.get()}")
    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"Recall over the last 100 instances: {metric.get():.3f}")
