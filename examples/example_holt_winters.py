"""Detect an outage in a series with a daily cycle with Holt-Winters aberrant behavior detection."""

import numpy as np

from pysad.models import HoltWinters

if __name__ == "__main__":
    period = 24  # hourly values with a daily cycle
    t = np.arange(30 * period)
    rng = np.random.default_rng(0)
    values = 10.0 + 5.0 * np.sin(2 * np.pi * t / period) + 0.5 * rng.normal(size=t.size)
    values[600:612] -= 4.0  # a 12-hour outage

    # A failure is at least 7 values outside the confidence band among the last 9.
    model = HoltWinters(period=period, window_size=9, threshold=7)
    scores = model.fit_score(values.reshape(-1, 1))

    # A score above 0 is a failure: the number of violations in the window.
    failures = np.flatnonzero(scores)
    print(f"Failures from {failures[0]} to {failures[-1]}, outage at [600, 612)")
