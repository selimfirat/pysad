"""Detect a change in the mean of a stream with the Page-Hinkley statistic."""

import numpy as np

from pysad.statistics import PageHinkleyMeter

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0, 1, 500), rng.normal(1, 1, 500)])

    # delta discounts small deviations; the threshold sets when to raise an alarm.
    meter = PageHinkleyMeter(delta=0.1)
    threshold = 50.0

    for index, value in enumerate(values):
        meter.update(value)
        # get() returns the two-sided statistic, so either direction can trigger an alarm.
        if meter.get() > threshold:
            print(f"First change detected at index {index} (zero-based)")
            break
