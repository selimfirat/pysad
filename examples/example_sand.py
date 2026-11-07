"""Find subsequences of an unusual shape in a periodic series with SAND."""

import numpy as np

from pysad.evaluation import AUROCMetric
from pysad.models import SAND

if __name__ == "__main__":
    length = 50  # the length of the anomalies, and of the subsequences that SAND scores
    rng = np.random.default_rng(0)
    values = np.sin(2 * np.pi * np.arange(6000) / length) + 0.05 * rng.normal(size=6000)
    labels = np.zeros(6000, dtype=int)
    for start in [2600, 3700, 4400, 5300]:
        values[start : start + length] = np.sin(4 * np.pi * np.arange(length) / length)
        # The score of a value is the score of the subsequence that ends with it.
        labels[start + length - 1 : start + 2 * length - 1] = 1

    # SAND clusters each batch of 2000 values and scores 0.0 until the first batch completes.
    model = SAND(subsequence_length=length, batch_size=2000, random_state=0)
    metric = AUROCMetric()

    for i, value in enumerate(values):
        score = model.fit_score_partial(np.array([value]))
        if i >= 2000:
            metric.update(labels[i], score)

    print(f"AUROC after the first batch: {metric.get():.3f}")
