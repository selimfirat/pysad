"""Detect a rise in the error rate of a stream of bits with the EDDM drift meter."""

import numpy as np

from pysad.statistics import EDDMMeter

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # A stream of errors (1) and correct predictions (0) whose error rate rises from 0.05 to 0.3.
    values = np.concatenate([rng.random(2000) < 0.05, rng.random(1000) < 0.3]).astype(float)

    meter = EDDMMeter()
    first_warning = None
    for index, value in enumerate(values):
        meter.update(value)
        if meter.warning_detected and first_warning is None:
            first_warning = index
        if meter.drift_detected:
            print(
                f"Warning at index {first_warning}, drift at index {index} (zero-based), "
                f"mean distance between errors {meter.get():.1f}"
            )
            first_warning = None
