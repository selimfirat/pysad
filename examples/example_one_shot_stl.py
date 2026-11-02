"""Remove the trend and the daily cycle with OneShotSTL, then score the residuals."""

import numpy as np

from pysad.evaluation import AUROCMetric
from pysad.models import StandardAbsoluteDeviation
from pysad.transform.preprocessing import OneShotSTL

if __name__ == "__main__":
    period = 24  # hourly values with a daily cycle
    t = np.arange(24 * 60)
    rng = np.random.default_rng(0)
    values = np.sin(2 * np.pi * t / period) + 0.1 * rng.normal(size=t.size)
    values[t >= 700] += 2.0  # the level jumps and stays there
    labels = np.zeros(t.size, dtype=int)
    spikes = rng.choice(np.arange(200, t.size), size=20, replace=False)
    labels[spikes] = 1
    values[spikes] += rng.choice([-0.8, 0.8], size=spikes.size)

    # OneShotSTL returns 0 for the first four periods, which estimate the cycle.
    decomposer = OneShotSTL(period=period)
    model = StandardAbsoluteDeviation()
    metric = AUROCMetric()

    for value, label in zip(values, labels, strict=True):
        residual = decomposer.fit_transform_partial(np.array([value]))
        score = model.fit_score_partial(residual)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
