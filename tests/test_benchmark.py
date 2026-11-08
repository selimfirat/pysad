"""The metrics of benchmarks/benchmark.py."""

import numpy as np
from sklearn.metrics import roc_auc_score

from tests.test_dimensionality import load_benchmark


def test_last_bit_differences_give_the_same_auroc():
    benchmark = load_benchmark()
    y = np.array([0, 1, 0, 1, 0, 0])
    scores = np.array([0.3, 0.3, 0.7, 0.7, 0.1, 1e-5])
    # another platform: the scores of the anomalies are one bit higher
    other = np.where(y == 1, np.nextafter(scores, np.inf), scores)
    assert roc_auc_score(y, scores) != roc_auc_score(y, other)
    assert roc_auc_score(y, benchmark.significant(scores)) == roc_auc_score(
        y, benchmark.significant(other)
    )
