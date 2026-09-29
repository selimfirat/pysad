# Import modules.
import numpy as np

from pysad.models import SeasonalESD, SeasonalHybridESD
from pysad.utils import ArrayStreamer

# This example demonstrates the usage of SeasonalESD and SeasonalHybridESD on a synthetic seasonal series.
if __name__ == "__main__":
    np.random.seed(61)  # Fix random seed.

    period = (
        24  # Number of observations in one seasonal period (e.g., hourly data with a daily cycle).
    )
    n_points = 400
    t = np.arange(n_points)
    X_all = np.sin(2 * np.pi * t / period) + 0.1 * np.random.randn(
        n_points
    )  # Seasonal series with noise.
    y_all = np.zeros(n_points)
    spikes = [150, 260, 340]  # Positions of the injected anomalies.
    X_all[spikes] += 4.0  # Inject spikes.
    y_all[spikes] = 1
    X_all = X_all.reshape(-1, 1)

    # window_size: number of recent observations used for STL decomposition and the ESD test.
    # max_anomalies: maximum number of anomalies tested per window.
    # Constructor constraints: window_size >= 2 * period and max_anomalies <= 0.49 * window_size.
    # SeasonalHybridESD uses median/MAD instead of mean/std, so it is more sensitive and may flag more points.
    models = {
        "SeasonalESD": SeasonalESD(period=period, window_size=100, max_anomalies=3, alpha=0.001),
        "SeasonalHybridESD": SeasonalHybridESD(
            period=period, window_size=100, max_anomalies=3, alpha=0.001
        ),
    }

    iterator = ArrayStreamer(shuffle=False)  # Create streamer to simulate streaming data.

    for name, model in models.items():
        detected = []
        for i, (X, y) in enumerate(iterator.iter(X_all, y_all)):  # Iterate over examples.
            score = model.fit_score_partial(X)  # Fit to the example and get its anomaly score.

            if score > 0:  # A positive score means the ESD test flagged the latest point.
                detected.append(i)

        # Output detected anomaly indices.
        print(name, "flagged indices:", detected, "| true anomalies:", spikes)
