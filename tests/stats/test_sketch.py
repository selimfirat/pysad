import pickle

import pytest

from pysad.statistics import CountMinSketchStatistic, HyperLogLogStatistic
from pysad.statistics.running_statistic import RunningStatistic

SEED = 42


def test_empty_statistics_return_zero():
    count_min = CountMinSketchStatistic()
    hyper_log_log = HyperLogLogStatistic()
    assert count_min.get() == 0.0
    assert hyper_log_log.get() == 0.0
    assert isinstance(count_min.get(), float)
    assert isinstance(hyper_log_log.get(), float)


def test_update_returns_self():
    count_min = CountMinSketchStatistic()
    hyper_log_log = HyperLogLogStatistic()
    assert count_min.update("a") is count_min
    assert hyper_log_log.update("a") is hyper_log_log


def test_count_min_sketch_estimates_recent_value():
    meter = CountMinSketchStatistic()
    for _ in range(7):
        meter.update("x")
    # Repeating one value hits the same counters, so the estimate is exact.
    assert meter.get() == 7.0

    meter.update("y")
    assert meter.get() >= 1.0


def test_count_min_sketch_never_underestimates():
    meter = CountMinSketchStatistic()
    for _ in range(5):
        meter.update("a")
    assert meter.get() >= 5.0
    for _ in range(2):
        meter.update("b")
    assert meter.get() >= 2.0


def test_count_min_sketch_numeric_values():
    meter = CountMinSketchStatistic()
    for _ in range(4):
        meter.update(1.5)
    assert meter.get() == 4.0


def test_count_min_sketch_invalid_width_and_depth_raise_value_error():
    with pytest.raises(ValueError, match="width must be a positive integer"):
        CountMinSketchStatistic(width=0)
    with pytest.raises(ValueError, match="width must be a positive integer"):
        CountMinSketchStatistic(width=-3)
    with pytest.raises(ValueError, match="depth must be a positive integer"):
        CountMinSketchStatistic(depth=0)


def test_count_min_sketch_random_state_reproducible():
    data = ["a", "b", "c", "a", "b", "a"]
    first = CountMinSketchStatistic(random_state=SEED)
    second = CountMinSketchStatistic(random_state=SEED)
    for value in data:
        first.update(value)
        second.update(value)
    assert first.get() == second.get()


def test_count_min_sketch_merge_combines_counts():
    first = CountMinSketchStatistic(random_state=SEED)
    second = CountMinSketchStatistic(random_state=SEED)
    for _ in range(5):
        first.update("a")
    for _ in range(3):
        second.update("a")
    for _ in range(2):
        second.update("b")
    first.merge(second)
    # second stays unchanged.
    assert second.get() >= 2.0
    first.update("a")
    # Merged sketch counts five plus three plus one occurrences of "a".
    assert first.get() >= 9.0


def test_count_min_sketch_merge_incompatible_raises():
    first = CountMinSketchStatistic(width=10, depth=2, random_state=SEED)
    second = CountMinSketchStatistic(width=20, depth=2, random_state=SEED)
    with pytest.raises(ValueError):
        first.merge(second)


def test_count_min_sketch_merge_invalid_type_raises_type_error():
    meter = CountMinSketchStatistic()
    with pytest.raises(TypeError, match="Cannot merge"):
        meter.merge("invalid_object")


def test_count_min_sketch_pickle_round_trip():
    meter = CountMinSketchStatistic(random_state=7)
    for value in ["a", "b", "a", "c", "a"]:
        meter.update(value)
    restored = pickle.loads(pickle.dumps(meter))
    assert restored.get() == meter.get()
    restored.update("a")
    meter.update("a")
    assert restored.get() == meter.get()


def test_hyperloglog_cardinality_estimate():
    meter = HyperLogLogStatistic()
    values = [f"item-{i}" for i in range(1_000)]
    for value in values:
        meter.update(value)
    # precision 14 has about 0.8% standard error, so 1000 distinct values stay near 1000.
    assert 900.0 <= meter.get() <= 1_100.0


def test_hyperloglog_duplicates_kept_low():
    meter = HyperLogLogStatistic()
    for _ in range(1_000):
        meter.update("only-one")
    # One distinct value stays near one and never grows to the stream length.
    assert 1.0 <= meter.get() <= 10.0


def test_hyperloglog_invalid_precision_raise_value_error():
    with pytest.raises(ValueError, match="precision must be between 4 and 18"):
        HyperLogLogStatistic(precision=3)
    with pytest.raises(ValueError, match="precision must be between 4 and 18"):
        HyperLogLogStatistic(precision=19)


def test_hyperloglog_merge_combines_cardinality():
    first = HyperLogLogStatistic()
    second = HyperLogLogStatistic()
    for i in range(500):
        first.update(f"a-{i}")
    for i in range(500):
        second.update(f"b-{i}")
    before_first = first.get()
    before_second = second.get()
    first.merge(second)
    assert first.get() >= before_first
    assert first.get() >= before_second
    assert first.get() <= 1_200.0
    # second stays unchanged.
    assert second.get() == before_second


def test_hyperloglog_merge_mismatched_precision_raises():
    first = HyperLogLogStatistic(precision=8)
    second = HyperLogLogStatistic(precision=12)
    with pytest.raises(ValueError):
        first.merge(second)


def test_hyperloglog_merge_invalid_type_raises_type_error():
    meter = HyperLogLogStatistic()
    with pytest.raises(TypeError, match="Cannot merge"):
        meter.merge("invalid_object")


def test_hyperloglog_pickle_round_trip():
    meter = HyperLogLogStatistic()
    for i in range(500):
        meter.update(f"item-{i}")
    restored = pickle.loads(pickle.dumps(meter))
    assert restored.get() == meter.get()
    restored.update("new-item")
    meter.update("new-item")
    assert restored.get() == meter.get()


def test_statistics_do_not_support_removal():
    for meter in (CountMinSketchStatistic(), HyperLogLogStatistic()):
        assert meter._removable is False
        with pytest.raises(TypeError, match="does not support remove"):
            RunningStatistic(type(meter), window_size=5)
