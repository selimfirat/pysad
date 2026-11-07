"""Detect a rise in the error rate of a stream of 0/1 errors with the DDM drift meter."""

import numpy as np

from pysad.statistics import DDMMeter

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # A stream of errors (1) and correct predictions (0) whose error rate rises from 0.2 to 0.5.
    errors = np.concatenate([rng.random(1000) < 0.2, rng.random(1000) < 0.5]).astype(float)

    meter = DDMMeter()
    # The start of the current warning zone. A warning that ends without a drift is a false alarm.
    warning_start = None
    for index, error in enumerate(errors):
        meter.update(error)
        if meter.drift_detected:
            print(f"Warning from index {warning_start}, drift at index {index} (zero-based)")
            break
        if not meter.warning_detected:
            warning_start = None
        elif warning_start is None:
            warning_start = index
