"""Accuracy and contract tests for :class:`pysad.utils.HyperLogLog`.

A HyperLogLog sketch must stay within a few percent of the true distinct count
while its memory stays ``2 ** precision`` bytes, treat repeats as one item, and
merge into the estimate of the union.
"""

import pytest

from pysad.utils import HyperLogLog

DISTINCT = 10_000


def test_estimates_a_large_stream_within_a_few_percent():
    sketch = HyperLogLog()
    sketch.update_many(range(DISTINCT))

    relative_error = abs(len(sketch) - DISTINCT) / DISTINCT
    assert relative_error < 0.03


def test_duplicates_count_once():
    once = HyperLogLog()
    once.update_many(range(1_000))

    repeated = HyperLogLog()
    for _ in range(5):
        repeated.update_many(range(1_000))

    assert abs(len(repeated) - len(once)) / len(once) < 0.05


def test_merge_estimates_the_union():
    left = HyperLogLog()
    left.update_many(range(2_000))
    right = HyperLogLog()
    right.update_many(range(2_000, 4_000))

    left.merge(right)

    assert abs(len(left) - 4_000) / 4_000 < 0.05


def test_merge_leaves_the_other_sketch_usable():
    left = HyperLogLog()
    left.update_many(range(500))
    right = HyperLogLog()
    right.update_many(range(500, 1_000))

    before = len(right)
    left.merge(right)

    assert len(right) == before


def test_merge_rejects_a_different_precision():
    with pytest.raises(ValueError):
        HyperLogLog(precision=10).merge(HyperLogLog(precision=12))


def test_empty_sketch_counts_nothing():
    sketch = HyperLogLog()

    assert len(sketch) == 0
    assert sketch.estimate() == 0.0


def test_memory_follows_the_precision_and_not_the_stream():
    sketch = HyperLogLog(precision=10)
    assert sketch.num_registers == 2**10

    sketch.update_many(range(50_000))

    assert sketch.num_registers == 2**10
    assert len(sketch) > 0


@pytest.mark.parametrize("precision", [3, 19])
def test_rejects_a_precision_outside_the_paper(precision):
    with pytest.raises(ValueError):
        HyperLogLog(precision=precision)
