"""Score a stream with DILOF, the local outlier factor in a window that keeps a summary of old points."""

from pysad.evaluation import AUROCMetric
from pysad.models import DILOF
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    # When the window holds 200 points, the model replaces the oldest 100 with a summary of 50.
    model = DILOF(num_neighbors=8, window_size=200)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
