import pickle

import numpy as np
import pytest

from pysad.statistics import KLLMeter

SEED = 42
NUM_POINTS = 10_000
QUANTILES = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]


def test_empty_meter_returns_zero():
    meter = KLLMeter()
    assert meter.num_items == 0
    assert meter.get() == 0.0
    assert meter.quantile(0.5) == 0.0
    assert meter.quantile(0.0) == 0.0
    assert meter.quantile(1.0) == 0.0


@pytest.mark.parametrize("q", [-0.01, 1.01, -1.0, 2.0])
def test_invalid_q_raises_value_error(q):
    with pytest.raises(ValueError, match="q must be in \\[0, 1\\]"):
        KLLMeter(q=q)

    meter = KLLMeter()
    with pytest.raises(ValueError, match="q must be in \\[0, 1\\]"):
        meter.quantile(q)


@pytest.mark.parametrize("k", [0, 1, 7, -5])
def test_invalid_k_raises_value_error(k):
    with pytest.raises(ValueError, match="k must be at least 8"):
        KLLMeter(k=k)


@pytest.mark.parametrize("epsilon", [0.0, 1.0, -0.1, 1.5])
def test_invalid_epsilon_raises_value_error(epsilon):
    with pytest.raises(ValueError, match="epsilon must be in \\(0, 1\\)"):
        KLLMeter(epsilon=epsilon)


def test_update_returns_self_and_increments_count():
    meter = KLLMeter()
    ret = meter.update(42.0)
    assert ret is meter
    assert meter.num_items == 1
    assert meter.get() == 42.0


@pytest.mark.parametrize("q", QUANTILES)
def test_uniform_stream_quantile_accuracy(q):
    rng = np.random.RandomState(SEED)
    data = rng.uniform(0.0, 100.0, size=NUM_POINTS)

    meter = KLLMeter(q=q, k=200)
    for val in data:
        meter.update(float(val))

    estimate = meter.get()
    exact = np.quantile(data, q)

    # With k=200, approximation error is well within 2% of the data range
    assert abs(estimate - exact) <= 2.5


def test_normal_stream_quantiles():
    rng = np.random.RandomState(SEED)
    data = rng.randn(NUM_POINTS)

    meter = KLLMeter(k=200)
    for val in data:
        meter.update(float(val))

    assert abs(meter.get() - 0.0) < 0.1
    for q in [0.1, 0.25, 0.75, 0.9]:
        est = meter.quantile(q)
        exact = np.quantile(data, q)
        assert abs(est - exact) < 0.15


def test_compaction_keeps_stored_items_bounded():
    rng = np.random.RandomState(SEED)
    data = rng.uniform(0.0, 1000.0, size=20_000)

    k = 100
    meter = KLLMeter(k=k)
    for val in data:
        meter.update(float(val))

    total_stored = sum(len(level) for level in meter._levels)
    assert total_stored <= 3 * k
    assert len(meter._levels) > 1
    assert meter.num_items == 20_000


def test_merge_combines_partitioned_streams():
    rng = np.random.RandomState(SEED)
    s1 = rng.uniform(0.0, 100.0, size=5000)
    s2 = rng.uniform(0.0, 100.0, size=5000)
    combined = np.concatenate([s1, s2])

    m1 = KLLMeter(q=0.5, k=200)
    for x in s1:
        m1.update(float(x))

    m2 = KLLMeter(q=0.5, k=200)
    for x in s2:
        m2.update(float(x))

    m2_before_items = m2.num_items
    m2_before_val = m2.get()

    m1.merge(m2)

    assert m1.num_items == 10_000
    assert abs(m1.get() - np.median(combined)) < 2.0

    # m2 must remain unchanged
    assert m2.num_items == m2_before_items
    assert m2.get() == m2_before_val


def test_merge_invalid_type_raises_type_error():
    meter = KLLMeter()
    with pytest.raises(TypeError, match="Cannot merge"):
        meter.merge("invalid_object")


def test_pickling_preserves_state_and_functionality():
    meter = KLLMeter(q=0.75, k=150)
    for i in range(500):
        meter.update(float(i))

    dumped = pickle.dumps(meter)
    restored = pickle.loads(dumped)

    assert restored.q == meter.q
    assert restored.k == meter.k
    assert restored.num_items == meter.num_items
    assert restored.get() == meter.get()

    meter.update(500.0)
    restored.update(500.0)
    assert restored.get() == meter.get()
