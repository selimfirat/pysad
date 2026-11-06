"""Detect a change in the mean of a stream with the Page-Hinkley statistic."""

import numpy as np

from pysad.statistics import PageHinkleyMeter

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    values = np.concatenate(
        [rng.normal(loc=0.0, size=500), rng.normal(loc=1.0, size=500)]
    )  # the mean moves from 0 to 1 at index 500

    # Deviations from the running mean smaller than delta do not accumulate.
    meter = PageHinkleyMeter(delta=0.5)
    threshold = 10.0  # raise an alarm when the two-sided cumulative sum is above it

    alarm = None
    for t, value in enumerate(values):
        if meter.update(value).get() > threshold:
            alarm = t
            break

    print(f"change detected at index {alarm}")
