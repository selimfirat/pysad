"""Track the mean, the variance and the 99th percentile of a stream with running statistics."""

import numpy as np

from pysad.statistics import AverageMeter, DDSketchMeter, VarianceMeter

if __name__ == "__main__":
    values = np.random.default_rng(0).normal(size=1000)  # mean 0, standard deviation 1

    mean = AverageMeter()
    variance = VarianceMeter()
    p99 = DDSketchMeter(q=0.99, relative_accuracy=0.01)  # within 1% of the exact value

    for value in values:
        mean.update(value)
        variance.update(value)
        p99.update(value)

    print(f"mean: {mean.get():.3f}, standard deviation: {np.sqrt(variance.get()):.3f}")
    print(
        f"99th percentile: {p99.get():.3f} (exact: {np.quantile(values, 0.99, method='lower'):.3f})"
    )
