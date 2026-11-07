"""Detect a fall in the accuracy of a stream of 0/1 errors with the FHDDM drift meter."""

import numpy as np

from pysad.statistics import FHDDMMeter

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # A stream of errors (1) and correct predictions (0) whose error rate rises from 0.2 to 0.5.
    errors = np.concatenate([rng.random(1000) < 0.2, rng.random(1000) < 0.5]).astype(float)

    meter = FHDDMMeter()
    for index, error in enumerate(errors):
        meter.update(error)
        if meter.drift_detected:
            print(f"Drift at index {index} (zero-based), error rate in the window {meter.get():.2f}")
            break
