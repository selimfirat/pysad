"""Score spikes in a seasonal series with SpectralResidual."""

import numpy as np
from sklearn.metrics import roc_auc_score

from pysad.models import SpectralResidual

if __name__ == "__main__":
    t = np.arange(2000)
    rng = np.random.default_rng(0)
    values = np.sin(2 * np.pi * t / 24) + 0.1 * rng.normal(size=t.size)
    labels = np.zeros(t.size, dtype=int)
    spikes = rng.choice(np.arange(100, t.size), size=40, replace=False)
    values[spikes] += rng.choice([-3.0, 3.0], size=spikes.size)
    labels[spikes] = 1

    # Each score compares the saliency of the newest value with that of the 21 values before it,
    # in a window of the last 30 values and 5 estimated values after them.
    model = SpectralResidual(window_size=30)
    scores = model.fit_score(values.reshape(-1, 1))

    # The scores are 0 until the window is full, so skip them.
    print(f"AUROC: {roc_auc_score(labels[100:], scores[100:]):.3f}")
