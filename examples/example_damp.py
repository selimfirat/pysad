"""Find the first unusual shape in a long series with DAMP, which searches only as far back as it needs."""

import numpy as np

from pysad.models import DAMP

if __name__ == "__main__":
    period = 50
    t = np.arange(20000)
    rng = np.random.default_rng(0)
    values = np.sin(2 * np.pi * t / period) + 0.1 * rng.normal(size=t.size)
    # One cycle at twice the frequency. Its values stay within the normal range of the sine.
    values[15000:15050] = np.sin(4 * np.pi * t[15000:15050] / period) + 0.1 * rng.normal(
        size=period
    )

    # The backward search can reach all earlier values, but most values need only the last cycles.
    model = DAMP(window_length=period, history_size=20000)
    scores = np.array([model.fit_score_partial(np.array([value])) for value in values])

    # The score of a value belongs to the subsequence of the last window_length values.
    end = scores.argmax() + 1
    print(f"Most unusual subsequence at [{end - period}, {end}), anomalous cycle at [15000, 15050)")
