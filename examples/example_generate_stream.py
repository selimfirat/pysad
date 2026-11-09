"""Score a synthetic stream with point, burst and drift anomalies."""

from pysad.evaluation import AUROCMetric
from pysad.models import LODA
from pysad.utils import ArrayStreamer, generate_stream

if __name__ == "__main__":
    X, y = generate_stream(random_state=0)

    model = LODA(random_state=0)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
