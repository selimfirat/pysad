"""Detect a rise in the rate of a stream of bits with the HDDM_A and HDDM_W drift meters."""

import numpy as np

from pysad.statistics import HDDMAMeter, HDDMWMeter

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # A stream of errors (1) and correct predictions (0) whose error rate rises from 0.2 to 0.5.
    values = np.concatenate([rng.random(1000) < 0.2, rng.random(1000) < 0.5]).astype(float)

    for meter in [HDDMAMeter(), HDDMWMeter()]:
        first_warning = None
        for index, value in enumerate(values):
            meter.update(value)
            if meter.warning_detected and first_warning is None:
                first_warning = index
            if meter.drift_detected:
                print(
                    f"{type(meter).__name__}: warning at index {first_warning}, "
                    f"drift at index {index} (zero-based)"
                )
                break
