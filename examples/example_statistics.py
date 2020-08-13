import numpy as np

from pysad.statistics import AverageMeter
from pysad.statistics import VarianceMeter

# This example shows the usage of statistics module for streaming data.
if __name__ == '__main__':

    X = np.random.randn(1000)
    avg_meter = AverageMeter()
    var_meter = VarianceMeter()

    for i in range(1000):
        avg_meter.update(X[i])
        var_meter.update(X[i])

    print(f"Average: {avg_meter.get()}, Standard deviation: {np.sqrt(var_meter.get())}")
    # It is close to random normal distribution with mean 0 and std 1 as we init the array via np.random.rand.
