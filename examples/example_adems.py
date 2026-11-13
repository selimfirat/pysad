"""Score a stream with ADeMS, the residual of each point against a Frequent Directions sketch."""

from pysad.evaluation import AUROCMetric
from pysad.models import ADeMS
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    # No threshold: the initial stream instances bootstrap the sketch.
    model = ADeMS()
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
