"""Evaluate a detector on a labeled stream with several metrics at once."""

from pysad.evaluation import AUPRMetric, AUROCMetric, WindowedMetric, evaluate_stream
from pysad.models import LODA
from pysad.utils import Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    results = evaluate_stream(
        model=LODA(random_state=0),
        X=X,
        y=y,
        metrics={
            "AUROC": AUROCMetric(),
            "AUPR": AUPRMetric(),
            "Windowed AUROC": WindowedMetric(metric_cls=AUROCMetric, window_size=100),
        },
    )

    for name, value in results.items():
        print(f"{name}: {value:.3f}")
