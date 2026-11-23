"""Compare detectors on a synthetic stream."""

import numpy as np

from pysad.evaluation import AUROCMetric
from pysad.models import (
    LODA,
    HalfSpaceTrees,
    RSForest,
    xStream,
)
from pysad.utils import ArrayStreamer, generate_stream


def evaluate_detector(name: str, model, X: np.ndarray, y: np.ndarray) -> float:
    metric = AUROCMetric()
    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)
    return metric.get()


if __name__ == "__main__":
    X, y = generate_stream(random_state=0)

    detectors = {
        "LODA": lambda: LODA(random_state=0),
        "xStream": lambda: xStream(random_state=0),
        "HalfSpaceTrees": lambda: HalfSpaceTrees(
            feature_mins=X.min(axis=0), feature_maxes=X.max(axis=0), random_state=0
        ),
        "RSForest": lambda: RSForest(random_state=0),
    }

    print("Comparing detectors on synthetic stream:")
    print(f"Stream: {X.shape[0]} instances, {X.shape[1]} features, {int(y.sum())} anomalies")
    print(
        "             LODA: AUROC {:.3f}".format(
            evaluate_detector("LODA", LODA(random_state=0), X, y)
        )
    )
    print(
        "          xStream: AUROC {:.3f}".format(
            evaluate_detector("xStream", xStream(random_state=0), X, y)
        )
    )
    print(
        "    HalfSpaceTrees: AUROC {:.3f}".format(
            evaluate_detector(
                "HalfSpaceTrees",
                HalfSpaceTrees(
                    feature_mins=X.min(axis=0), feature_maxes=X.max(axis=0), random_state=0
                ),
                X,
                y,
            )
        )
    )
    print(
        "          RSForest: AUROC {:.3f}".format(
            evaluate_detector("RSForest", RSForest(random_state=0), X, y)
        )
    )
