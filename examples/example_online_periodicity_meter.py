"""Estimate the period of a periodic stream with an online autocorrelation meter."""

import numpy as np

from pysad.statistics import OnlinePeriodicityMeter

if __name__ == "__main__":
    # A sine wave with period 24. Its autocorrelation is strongest at lag 24.
    period = 24
    meter = OnlinePeriodicityMeter(max_lag=60, alpha=0.05)

    for t in range(5000):
        value = np.sin(2 * np.pi * t / period)
        meter.update(value)

    print(f"estimated period: {meter.get():.0f} (true period: {period})")
