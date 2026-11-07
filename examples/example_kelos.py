"""Score a stream with KELOS, the kernel density local outliers over micro-clusters of a window."""

from pysad.evaluation import AUROCMetric
from pysad.models import KELOS
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    # The default cluster radius fits features scaled to [0, 1].
    X, y = Data().get_stream("arrhythmia.mat", random_state=0, scale=True)

    # The stream has 452 points, so a window of 200 slides through it.
    model = KELOS(window_size=200)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
