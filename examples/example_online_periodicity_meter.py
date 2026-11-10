"""Estimate a stream period with the original OnlineSLE algorithm."""

import numpy as np

from pysad.statistics import OnlinePeriodicityMeter

if __name__ == "__main__":
    period = 24
    meter = OnlinePeriodicityMeter(window_size=240)

    for t in range(1000):
        value = np.sin(2 * np.pi * t / period)
        meter.update(value)

    print(f"estimated period: {meter.get():.0f} (true period: {period})")
