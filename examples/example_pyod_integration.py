"""Run a batch PyOD Isolation Forest on a stream with a sliding reference window."""

from pyod.models.iforest import IForest

from pysad.evaluation import AUROCMetric
from pysad.models.integrations import ReferenceWindowModel
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    # Train on the first 100 points, then retrain on the last 240 points every 30 points.
    model = ReferenceWindowModel(
        IForest,
        window_size=240,
        sliding_size=30,
        initial_window_X=X[:100],
        random_state=61,
    )
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X[100:], y[100:]):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
