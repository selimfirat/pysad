"""Find a level shift in a noisy series with ChangeFinder."""

import numpy as np

from pysad.models import ChangeFinder

if __name__ == "__main__":
    t = np.arange(1000)
    rng = np.random.default_rng(0)
    values = 0.5 + 0.2 * np.sin(2 * np.pi * t / 50) + 0.02 * rng.normal(size=t.size)
    values[600:] += 0.3  # a level shift

    # A larger discount forgets old values faster. change_point=False gives the outlier score.
    model = ChangeFinder(order=1, discount=0.01, window_size=5)
    scores = model.fit_score(values.reshape(-1, 1))

    # The first scores are high while the models learn the series, so skip them.
    print(f"Highest change point score at {scores[100:].argmax() + 100}, level shift at 600")
