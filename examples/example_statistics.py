# Import modules.
from pysad.statistics import AverageMeter
from pysad.statistics import VarianceMeter
import numpy as np

# This example shows the usage of statistics module for streaming data.
if __name__ == '__main__':

    # Init data with mean 0 and standard deviation 1.
    X = np.random.randn(1000)

    # Init statistics trackers for mean and variance.
    avg_meter = AverageMeter()
    var_meter = VarianceMeter()

    for i in range(1000):
        # Update statistics trackers.
        avg_meter.update(X[i])
        var_meter.update(X[i])

    # Output resulting statistics.
    print(f"Average: {avg_meter.get()}, Standard deviation: {np.sqrt(var_meter.get())}")
    # It is close to random normal distribution with mean 0 and std 1 as we init the array via np.random.rand.
