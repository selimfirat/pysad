"""Find an unusual shape in a series with MatrixProfile, although each of its values is normal."""

import numpy as np

from pysad.models import MatrixProfile

if __name__ == "__main__":
    period = 50
    t = np.arange(2000)
    rng = np.random.default_rng(0)
    values = np.sin(2 * np.pi * t / period) + 0.1 * rng.normal(size=t.size)
    # One cycle at twice the frequency. Its values stay within the normal range of the sine.
    values[1200:1250] = np.sin(4 * np.pi * t[1200:1250] / period) + 0.1 * rng.normal(size=period)

    # Each score is the distance from the last window_length values to the most similar earlier
    # subsequence among the last history_size values.
    model = MatrixProfile(window_length=period, history_size=500)
    scores = np.array([model.fit_score_partial(np.array([value])) for value in values])

    # The first scores compare each subsequence with only a few earlier ones, so skip them.
    print(f"Highest score at {500 + scores[500:].argmax()}, anomalous cycle at [1200, 1250)")
