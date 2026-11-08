"""Stream a Polars DataFrame through PySAD and print anomaly alerts."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pysad.models import LODA
from pysad.utils import PolarsStreamer

if __name__ == "__main__":
    np.random.seed(42)

    num_rows = 200
    cpu = np.random.normal(loc=45, scale=4, size=num_rows)
    latency_ms = np.random.normal(loc=100, scale=8, size=num_rows)
    requests_per_minute = np.random.normal(loc=220, scale=15, size=num_rows)

    spike_positions = [80, 130, 180]
    cpu[spike_positions] += [40, 50, 45]
    latency_ms[spike_positions] += [120, 150, 130]

    dataframe = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=datetime(2025, 1, 1, 9),
                end=datetime(2025, 1, 1, 9) + timedelta(minutes=num_rows - 1),
                interval="1m",
                eager=True,
            ),
            "cpu": cpu,
            "latency_ms": latency_ms,
            "requests_per_minute": requests_per_minute,
        }
    )

    # A Polars DataFrame has no index, so the timestamps stay in a column outside the features.
    features = dataframe.drop("timestamp")

    model = LODA()
    scores = [model.fit_score_partial(row) for row in PolarsStreamer().iter(features)]

    highest_scoring_rows = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:5]

    print("Highest-scoring anomaly alerts:")
    for row_number, score in highest_scoring_rows:
        print(f"{dataframe['timestamp'][row_number].isoformat()} score={score:.4f}")
