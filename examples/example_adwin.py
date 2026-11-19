"""Track a changing mean with the adaptive window of ADWINMeter."""

import numpy as np

from pysad.statistics import ADWINMeter

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # ADWIN assumes values in [0, 1]. The mean changes from 0.25 to 0.75.
    values = np.concatenate([rng.uniform(0, 0.5, 1000), rng.uniform(0.5, 1, 1000)])

    meter = ADWINMeter()

    for index, value in enumerate(values):
        meter.update(value)
        if meter.drift_detected:
            print(
                f"Drift at index {index} (zero-based), "
                f"window size {meter.width}, mean {meter.get():.3f}"
            )

    print(f"Final window size {meter.width}, mean {meter.get():.3f}")
