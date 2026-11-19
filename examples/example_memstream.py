"""Train MemStream on known normal records, then score and update its bounded memory."""

from pysad.evaluation import AUROCMetric
from pysad.models import MemStream
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("ionosphere.mat")
    model = MemStream(window_size=16, random_state=0)
    # Supply a separate known-normal calibration block before monitoring.
    initial_X = X[:100][y[:100] == 0][: model.window_size]
    model.fit(initial_X)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X[100:], y[100:]):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
