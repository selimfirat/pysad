"""Score a stream with AnyOut, which reads a ClusTree down to the depth that its time budget allows."""

from pysad.evaluation import AUROCMetric
from pysad.models import AnyOut
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_stream("arrhythmia.mat", random_state=0, scale=True)

    # Each point descends 4 levels of the tree before it is scored, as if time ran out there.
    model = AnyOut(depth=4)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
