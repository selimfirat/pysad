"""Merge quantile summaries from partitions without retaining their observations."""

import numpy as np

from pysad.statistics import DDSketchMeter

if __name__ == "__main__":
    values = np.random.default_rng(0).lognormal(sigma=2.0, size=10_000)
    p99 = DDSketchMeter(q=0.99)

    for partition in np.array_split(values, 4):
        summary = DDSketchMeter()
        for value in partition:
            summary.update(float(value))
        p99.merge(summary)

    exact = np.quantile(values, 0.99, method="lower")
    print(f"Observations: {p99.num_items}")
    print(f"99th percentile: {p99.get():.3f} (exact: {exact:.3f})")
