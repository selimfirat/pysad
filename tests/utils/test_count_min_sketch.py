"""Accuracy and contract tests for :class:`pysad.utils.CountMinSketch`.

A count-min sketch must never underestimate a frequency, must stay within the
paper's ``epsilon * total`` bound, and must merge sketches built with the
same hash functions.
"""

import math

import numpy as np
import pytest

from pysad.utils import CountMinSketch


def _zipf_stream(seed, size=20_000):
    rng = np.random.RandomState(seed)
    return [int(x) for x in rng.zipf(1.3, size=size) % 5_000]


def test_never_underestimates():
    stream = _zipf_stream(0)
    sketch = CountMinSketch(width=200, depth=4, random_state=0)
    sketch.update_many(stream)

    values, counts = np.unique(stream, return_counts=True)
    for value, count in zip(values, counts, strict=True):
        assert sketch.estimate(int(value)) >= count


def test_estimates_stay_within_the_error_bound():
    stream = _zipf_stream(1)
    sketch = CountMinSketch.from_error_bound(epsilon=0.01, delta=0.01, random_state=1)
    sketch.update_many(stream)

    values, counts = np.unique(stream, return_counts=True)
    bound = sketch.epsilon * sketch.total
    over = [sketch.estimate(int(v)) - c for v, c in zip(values, counts, strict=True)]
    # Theorem 1: each estimate exceeds the bound with probability at most delta.
    assert np.mean(np.array(over) > bound) <= 0.01


def test_exact_when_items_do_not_collide():
    sketch = CountMinSketch(width=10_000, depth=5, random_state=0)
    for item, count in [("a", 3), ("b", 7), ("c", 1)]:
        sketch.update(item, count)

    assert sketch.estimate("a") == 3
    assert sketch.estimate("b") == 7
    assert sketch.estimate("c") == 1
    assert sketch.estimate("missing") == 0
    assert sketch.total == 11


def test_repeated_items_add_up():
    one_by_one = CountMinSketch(width=1_000, depth=4, random_state=0)
    one_by_one.update_many(["x"] * 5 + ["y"] * 2)
    batched = CountMinSketch(width=1_000, depth=4, random_state=0)
    batched.update("x", 5)
    batched.update("y", 2)

    assert one_by_one.estimate("x") == batched.estimate("x") == 5
    assert one_by_one.estimate("y") == batched.estimate("y") == 2


def test_from_error_bound_follows_the_paper():
    sketch = CountMinSketch.from_error_bound(epsilon=0.001, delta=0.01)

    assert sketch.width == math.ceil(math.e / 0.001)
    assert sketch.depth == math.ceil(math.log(1 / 0.01))


def test_merge_matches_one_sketch_over_both_streams():
    left_stream = _zipf_stream(2, size=5_000)
    right_stream = _zipf_stream(3, size=5_000)

    left = CountMinSketch(width=128, depth=4, random_state=7)
    left.update_many(left_stream)
    right = CountMinSketch(width=128, depth=4, random_state=7)
    right.update_many(right_stream)
    both = CountMinSketch(width=128, depth=4, random_state=7)
    both.update_many(left_stream + right_stream)

    left.merge(right)

    for item in set(left_stream + right_stream):
        assert left.estimate(item) == both.estimate(item)
    assert left.total == both.total


def test_merge_leaves_the_other_sketch_unchanged():
    left = CountMinSketch(width=32, depth=2, random_state=0)
    right = CountMinSketch(width=32, depth=2, random_state=0)
    right.update_many(range(100))
    before = [right.estimate(item) for item in range(100)]

    left.update_many(range(50))
    left.merge(right)

    assert [right.estimate(item) for item in range(100)] == before
    assert right.total == 100


def test_merge_rejects_a_different_shape():
    with pytest.raises(ValueError):
        CountMinSketch(width=32, depth=2, random_state=0).merge(
            CountMinSketch(width=64, depth=2, random_state=0)
        )


def test_merge_rejects_different_hash_functions():
    with pytest.raises(ValueError):
        CountMinSketch(width=32, depth=2, random_state=0).merge(
            CountMinSketch(width=32, depth=2, random_state=1)
        )


def test_same_seed_gives_the_same_sketch():
    first = CountMinSketch(width=50, depth=3, random_state=5)
    second = CountMinSketch(width=50, depth=3, random_state=5)
    first.update_many(range(1_000))
    second.update_many(range(1_000))

    assert [first.estimate(i) for i in range(1_000)] == [second.estimate(i) for i in range(1_000)]


def test_rejects_a_negative_count():
    with pytest.raises(ValueError):
        CountMinSketch().update("a", -1)


@pytest.mark.parametrize("kwargs", [{"width": 0}, {"depth": 0}])
def test_rejects_an_empty_shape(kwargs):
    with pytest.raises(ValueError):
        CountMinSketch(**kwargs)


@pytest.mark.parametrize("epsilon, delta", [(0, 0.1), (1, 0.1), (0.1, 0), (0.1, 1)])
def test_from_error_bound_rejects_values_outside_zero_one(epsilon, delta):
    with pytest.raises(ValueError):
        CountMinSketch.from_error_bound(epsilon, delta)
