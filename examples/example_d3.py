"""Score the readings of one temperature sensor with D3, which counts the neighbors of a reading in a kernel density estimate of its window."""

from pysad.evaluation import AUROCMetric
from pysad.models import D3
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("nab/realKnownCause/machine_temperature_system_failure.csv")
    # The default radius fits readings scaled to [0, 1], as in the paper.
    X = (X - X.min()) / (X.max() - X.min())

    model = D3(window_size=1000, random_state=0)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
