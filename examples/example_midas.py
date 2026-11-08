"""Detect a burst of repeated edges in a small graph stream with MIDAS."""

import numpy as np

from pysad.models import MIDAS
from pysad.utils import ArrayStreamer

if __name__ == "__main__":
    # Each row is [source, destination]: integer node IDs, not numeric measurements.
    regular_edges = np.tile([[0, 1], [1, 2], [2, 3], [3, 0]], (20, 1))
    tick_size = 4
    burst_edges = np.tile([[0, 1]], (tick_size, 1))
    edges = np.vstack([regular_edges, burst_edges])

    # PySAD groups every tick_size edges into a tick; it does not read timestamps.
    # The regular ticks contain each edge once; the final tick repeats one edge.
    model = MIDAS(tick_size=tick_size, random_state=0)
    scores = []

    for edge in ArrayStreamer().iter(edges):
        scores.append(model.fit_score_partial(edge))

    print(f"Maximum regular score: {max(scores[: len(regular_edges)]):.3f}")
    print(f"Last burst edge score: {scores[-1]:.3f}")
