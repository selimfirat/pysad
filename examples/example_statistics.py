"""Track the mean and the variance of a stream with running statistics."""

import numpy as np

from pysad.statistics import AverageMeter, VarianceMeter

if __name__ == "__main__":
    values = np.random.default_rng(0).normal(size=1000)  # mean 0, standard deviation 1

    mean = AverageMeter()
    variance = VarianceMeter()

    for value in values:
        mean.update(value)
        variance.update(value)

    print(f"mean: {mean.get():.3f}, standard deviation: {np.sqrt(variance.get()):.3f}")
