"""Score a stream with TEDA, the eccentricity of each point from the running mean and variance."""

from pysad.evaluation import AUROCMetric
from pysad.models import TEDA
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    model = TEDA()
    metric = AUROCMetric()
    num_alerts = 0

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        num_alerts += score > 3**2  # the paper's "m sigma" rule with m = 3
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
    print(f"Alerts: {num_alerts}")
