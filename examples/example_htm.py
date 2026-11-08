"""Detect anomalies in a timestamped series with HTM, the detector of Numenta in the Numenta Anomaly Benchmark (NAB)."""

import numpy as np

from pysad.models import HTM
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    # Each point is [timestamp, value], with the timestamp in seconds since the Unix epoch.
    X, y = Data().get_data("nab/realKnownCause/nyc_taxi.csv", timestamps=True)

    # As NAB does, give HTM the range of the values.
    model = HTM(min_val=X[:, 1].min(), max_val=X[:, 1].max(), random_state=0)
    likelihoods = np.array([model.fit_score_partial(x) for x in ArrayStreamer().iter(X)])

    # The paper reports an anomaly when the likelihood is at least 1 - 1e-5.
    alerts = np.flatnonzero(likelihoods >= 1 - 1e-5)
    num_windows = int((np.diff(np.r_[0, y]) == 1).sum())
    print(
        f"{len(alerts)} alerts, {int(y[alerts].sum())} of them in the {num_windows} anomaly windows"
    )
