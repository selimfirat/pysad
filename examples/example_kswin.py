"""Detect a change in the spread of a stream with the KSWIN drift meter."""

import numpy as np

from pysad.statistics import KSWINMeter

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # The mean stays 0, but the standard deviation changes from 1 to 10.
    values = np.concatenate([rng.normal(0, 1, 1000), rng.normal(0, 10, 1000)])

    meter = KSWINMeter(random_state=0)

    for index, value in enumerate(values):
        meter.update(value)
        if meter.drift_detected:
            print(f"Drift detected at index {index} (zero-based), distance {meter.get():.2f}")
