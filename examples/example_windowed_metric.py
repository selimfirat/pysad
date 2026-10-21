"""Evaluate a detector per window and over the whole stream."""

from pysad.evaluation import AUROCMetric, WindowedMetric
from pysad.models import xStream
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    model = xStream(random_state=0)
    metric = AUROCMetric()
    windowed_metric = WindowedMetric(metric_cls=AUROCMetric, window_size=100)

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)
        windowed_metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
    print(f"Windowed AUROC: {windowed_metric.get():.3f}")
