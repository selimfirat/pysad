"""Stream a pandas DataFrame through PySAD and print anomaly alerts."""

# Pandas is optional. Install it with: pip install pysad[pandas]

import numpy as np
import pandas as pd

from pysad.models import LODA
from pysad.utils import PandasStreamer

np.random.seed(42)

timestamps = pd.date_range("2025-01-01 09:00:00", periods=200, freq="min")
dataframe = pd.DataFrame(
    {
        "cpu": np.random.normal(loc=45, scale=4, size=len(timestamps)),
        "latency_ms": np.random.normal(loc=100, scale=8, size=len(timestamps)),
        "requests_per_minute": np.random.normal(loc=220, scale=15, size=len(timestamps)),
    },
    index=timestamps,
)

spike_positions = [80, 130, 180]
dataframe.iloc[spike_positions, dataframe.columns.get_loc("cpu")] += [40, 50, 45]
dataframe.iloc[spike_positions, dataframe.columns.get_loc("latency_ms")] += [120, 150, 130]

streamer = PandasStreamer(shuffle=False)
model = LODA()

scores = []
for row in streamer.iter(dataframe):
    scores.append(model.fit_score_partial(row))

highest_scoring_rows = sorted(
    enumerate(scores),
    key=lambda item: item[1],
    reverse=True,
)[:5]

print("Highest-scoring anomaly alerts:")
for row_number, score in highest_scoring_rows:
    print(f"{dataframe.index[row_number].isoformat()} score={score:.4f}")
