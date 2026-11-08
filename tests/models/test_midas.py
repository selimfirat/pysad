from collections import Counter

import numpy as np
import pytest


def _exact_scores(edges, tick_size):
    """The scores of Definition 1 of the paper, from exact counts instead of count-min sketches."""
    total, current, scores = Counter(), Counter(), []
    for i, edge in enumerate(map(tuple, edges)):
        t = i // tick_size + 1
        if i % tick_size == 0:
            current.clear()
        current[edge] += 1
        total[edge] += 1
        a, s = current[edge], total[edge]
        scores.append(0.0 if t == 1 else (a - s / t) ** 2 * t * t / (s * (t - 1)))
    return np.array(scores)


@pytest.mark.parametrize("tick_size", [1, 3, 10])
def test_scores_match_the_chi_squared_statistic_of_exact_counts(tick_size):
    from pysad.models import MIDAS

    # Few distinct edges and many buckets, so the sketches count every edge exactly.
    edges = np.random.RandomState(0).randint(0, 4, size=(200, 2)).astype(np.float64)

    scores = MIDAS(num_buckets=100003, tick_size=tick_size, random_state=0).fit_score(edges)

    np.testing.assert_allclose(scores, _exact_scores(edges, tick_size))


def test_burst_of_one_edge_gets_the_highest_scores():
    from pysad.models import MIDAS

    rng = np.random.RandomState(0)
    tick_size = 20
    normal = rng.randint(0, 10, size=(30 * tick_size, 2)).astype(np.float64)
    burst = np.tile([[3.0, 7.0]], (tick_size, 1))

    scores = MIDAS(tick_size=tick_size, random_state=0).fit_score(np.vstack([normal, burst]))

    # The score of the burst edge grows with its count in the tick.
    assert scores[len(normal) + tick_size // 2 :].min() > scores[: len(normal)].max()


def test_edge_needs_a_source_and_a_destination():
    from pysad.models import MIDAS

    with pytest.raises(ValueError, match=r"\[source, destination\]"):
        MIDAS().fit_partial(np.array([1.0, 2.0, 3.0]))
