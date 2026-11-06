"""Score a stream with RS-Forest, a density estimate from random space trees."""

from pysad.evaluation import AUROCMetric
from pysad.models import RSForest
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    # The first window sets the feature ranges of the trees and scores 0.0.
    # Each later point is scored against the density of the previous window.
    model = RSForest(window_size=64, random_state=0)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
