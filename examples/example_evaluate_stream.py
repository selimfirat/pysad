"""Evaluate a detector from single-use generators with several metrics at once."""

from pysad.evaluation import AUPRMetric, AUROCMetric, F1Metric, WindowedMetric, evaluate_stream
from pysad.models import LODA
from pysad.utils import Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    results = evaluate_stream(
        model=LODA(random_state=0),
        X=(x for x in X),
        y=(int(label) for label in y),
        metrics={
            "AUROC": AUROCMetric(),
            "AUPR": AUPRMetric(),
            # Illustrative raw-score cutoff; choose a threshold on prior validation data.
            "F1": F1Metric(threshold=3.0),
            "Windowed AUROC": WindowedMetric(metric_cls=AUROCMetric, window_size=100),
        },
    )

    for name, value in results.items():
        print(f"{name}: {value:.3f}")
