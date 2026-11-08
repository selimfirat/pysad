"""Score a stream with IncrementalLOF, the local outlier factor in a sliding window."""

from pysad.evaluation import AUROCMetric
from pysad.models import IncrementalLOF
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    model = IncrementalLOF(num_neighbors=10, window_size=1000)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
