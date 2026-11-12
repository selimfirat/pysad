"""Score spikes in a seasonal series by the forecast errors of a River forecaster. River needs Python 3.11 or newer."""

import numpy as np
from river import time_series
from sklearn.metrics import roc_auc_score

from pysad.integrations import PredictiveAnomalyDetector

if __name__ == "__main__":
    t = np.arange(2000)
    rng = np.random.default_rng(0)
    values = np.sin(2 * np.pi * t / 24) + 0.1 * rng.normal(size=t.size)
    labels = np.zeros(t.size, dtype=int)
    spikes = rng.choice(np.arange(100, t.size), size=40, replace=False)
    values[spikes] += rng.choice([-1.0, 1.0], size=spikes.size)
    labels[spikes] = 1

    # An autoregressive forecaster that predicts each value from the last 24 values.
    forecaster = time_series.SNARIMAX(p=24, d=0, q=0)
    model = PredictiveAnomalyDetector(forecaster, probationary_period=100)
    scores = model.fit_score(values.reshape(-1, 1))

    # The scores are 0 during the probationary period, so skip them.
    print(f"AUROC: {roc_auc_score(labels[100:], scores[100:]):.3f}")
