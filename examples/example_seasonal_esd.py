"""Find spikes in a series with a daily cycle with Seasonal ESD and Seasonal Hybrid ESD."""

import numpy as np

from pysad.models import SeasonalESD, SeasonalHybridESD

if __name__ == "__main__":
    period = 24  # hourly values with a daily cycle
    t = np.arange(400)
    rng = np.random.default_rng(0)
    values = np.sin(2 * np.pi * t / period) + 0.1 * rng.normal(size=t.size)
    spikes = [150, 260, 340]
    values[spikes] += 4.0

    # window_size must be greater than 2 * period. By default, the test flags only high values.
    # SeasonalHybridESD uses the median and the MAD, so earlier anomalies in the window affect it less.
    models = {
        "SeasonalESD": SeasonalESD(period=period, window_size=100, alpha=0.001),
        "SeasonalHybridESD": SeasonalHybridESD(period=period, window_size=100, alpha=0.001),
    }

    for name, model in models.items():
        # A score above 0 means that the test flagged the newest value.
        flagged = [
            i for i, value in enumerate(values) if model.fit_score_partial(np.array([value])) > 0
        ]
        print(f"{name} flagged {flagged}, true anomalies {spikes}")
