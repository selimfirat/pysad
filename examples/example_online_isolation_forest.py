"""Score a stream with Online Isolation Forest, an isolation forest that learns and forgets points in a bounded window."""

from pysad.evaluation import AUROCMetric
from pysad.models import OnlineIsolationForest
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    # The forest keeps the most recent 256 points and scores each point before it learns it.
    model = OnlineIsolationForest(num_trees=20, max_leaf_samples=8, window_size=256, random_state=0)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
