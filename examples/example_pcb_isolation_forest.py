"""Score a stream with PCB-iForest, the performance counter-based isolation forest."""

from pysad.evaluation import AUROCMetric
from pysad.models import PCBIsolationForest
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    # The first window builds the forest and scores 0.0. Each later point is scored
    # against the forest of the current window. An NDKSWIN drift test partially
    # refits the forest when the stream distribution changes.
    model = PCBIsolationForest(num_trees=50, window_size=200, random_state=0)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
