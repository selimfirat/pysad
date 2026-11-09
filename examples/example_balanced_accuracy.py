"""Measure balanced accuracy with a threshold on streaming anomaly scores."""

from pysad.evaluation import BalancedAccuracyMetric
from pysad.models import StandardAbsoluteDeviation
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")
    # Use the first feature for this univariate detector.
    X = X[:, :1]
    model = StandardAbsoluteDeviation()
    metric = BalancedAccuracyMetric(threshold=3.0)

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"Balanced accuracy: {metric.get():.3f}")
