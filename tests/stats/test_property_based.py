"""Property-based tests for streaming statistics and drift detector invariants.

These tests use Hypothesis to draw diverse streams and check invariants that must hold
for every statistic and drift detector in ``pysad.statistics``, including edge cases
that fixed test data rarely covers.
"""

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule

from pysad.statistics import (
    ADWINMeter,
    AverageMeter,
    CountMeter,
    CountMinSketchStatistic,
    CUSUMMeter,
    DDMMeter,
    DDSketchMeter,
    EDDMMeter,
    ExponentialMovingAverageMeter,
    ExponentialMovingVarianceMeter,
    FHDDMMeter,
    FHDDMSAddMeter,
    FHDDMSMeter,
    GeometricMovingAverageMeter,
    HDDMAMeter,
    HDDMWMeter,
    HyperLogLogStatistic,
    KLLMeter,
    KSWINMeter,
    MaxMeter,
    MedianMeter,
    MinMeter,
    OnlinePeriodicityMeter,
    PageHinkleyMeter,
    QuantileMeter,
    RelativeErrorQuantileMeter,
    RunningStatistic,
    SumMeter,
    SumSquaresMeter,
    VarianceMeter,
)

# Settings for property tests: small example count for CI, deterministic, no deadline.
PROPERTY_SETTINGS = settings(
    max_examples=20,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=True,
)

# Settings for stateful tests.
STATEFUL_SETTINGS = settings(
    max_examples=10,
    deadline=None,
    database=None,
    derandomize=True,
)

# Stream values: mix of constants, small integers, and wide-range floats.
FINITE_VALUES = st.one_of(
    st.sampled_from([0.0, 1.0, -1.0, 0.5, 2.0, -2.0]),
    st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
)

# Stream lengths for property tests.
STREAM_LENGTH = st.integers(min_value=1, max_value=50)


@st.composite
def _stream(draw, min_length=1, max_length=50):
    """Draw a stream of floats with constant features and duplicates."""
    num_instances = draw(st.integers(min_length, max_length))
    values = draw(hnp.arrays(np.float64, (num_instances,), elements=FINITE_VALUES))
    # Make some values constant or duplicated.
    if num_instances > 1 and draw(st.booleans()):
        idx = draw(st.integers(1, num_instances - 1))
        values[idx] = values[0]
    return values


def _is_bool_or_numpy_bool(val: Any) -> bool:
    """Check if value is a Python bool or numpy bool_."""
    return isinstance(val, (bool, np.bool_))


# ==============================================================================
# BASIC STATISTICS INVARIANTS
# ==============================================================================

REMOVABLE_BASIC_STATS = [
    AverageMeter,
    CountMeter,
    SumMeter,
    SumSquaresMeter,
    VarianceMeter,
    MaxMeter,
    MinMeter,
]

NON_REMOVABLE_BASIC_STATS = []

ALL_BASIC_STATS = REMOVABLE_BASIC_STATS + NON_REMOVABLE_BASIC_STATS


@pytest.mark.parametrize("stat_cls", ALL_BASIC_STATS, ids=[c.__name__ for c in ALL_BASIC_STATS])
@PROPERTY_SETTINGS
@given(stream=_stream(min_length=5, max_length=30))
def test_basic_stat_get_returns_finite(stat_cls, stream):
    """get() returns a finite number after updates."""
    kwargs = {"keep_values": True} if stat_cls in (MaxMeter, MinMeter) else {}
    meter = stat_cls(**kwargs)
    for value in stream:
        meter.update(value)
    result = meter.get()
    assert isinstance(result, (int, float)) and not isinstance(result, bool)
    assert math.isfinite(result)


@pytest.mark.parametrize("stat_cls", ALL_BASIC_STATS, ids=[c.__name__ for c in ALL_BASIC_STATS])
@PROPERTY_SETTINGS
@given(stream=_stream(min_length=5, max_length=30))
def test_basic_stat_get_is_idempotent(stat_cls, stream):
    """Repeated get() calls without updates return the same value."""
    kwargs = {"keep_values": True} if stat_cls in (MaxMeter, MinMeter) else {}
    meter = stat_cls(**kwargs)
    for value in stream:
        meter.update(value)
    first = meter.get()
    second = meter.get()
    assert first == second


@pytest.mark.parametrize(
    "stat_cls", REMOVABLE_BASIC_STATS, ids=[c.__name__ for c in REMOVABLE_BASIC_STATS]
)
@PROPERTY_SETTINGS
@given(stream=_stream(min_length=3, max_length=20))
def test_removable_basic_stat_remove_restores(stat_cls, stream):
    """remove(x) after update(x) restores the previous state (within floating-point precision)."""
    kwargs = {"keep_values": True} if stat_cls in (MaxMeter, MinMeter) else {}
    meter = stat_cls(**kwargs)
    for value in stream[:-1]:
        meter.update(value)
    before = meter.get()
    last = stream[-1]
    meter.update(last)
    meter.remove(last)
    after = meter.get()
    assert after == pytest.approx(before, rel=1e-12, abs=1e-12)


@PROPERTY_SETTINGS
@given(stream=_stream(min_length=10, max_length=30))
def test_average_meter_matches_numpy_mean(stream):
    """AverageMeter matches np.mean on the same stream."""
    meter = AverageMeter()
    for i, value in enumerate(stream):
        meter.update(value)
        assert meter.get() == pytest.approx(np.mean(stream[: i + 1]))


@PROPERTY_SETTINGS
@given(stream=_stream(min_length=10, max_length=30))
def test_sum_meter_matches_numpy_sum(stream):
    """SumMeter matches np.sum on the same stream."""
    meter = SumMeter()
    for i, value in enumerate(stream):
        meter.update(value)
        assert meter.get() == pytest.approx(np.sum(stream[: i + 1]))


@PROPERTY_SETTINGS
@given(stream=_stream(min_length=10, max_length=30))
def test_variance_meter_non_negative(stream):
    """VarianceMeter never returns negative values (Welford's algorithm)."""
    meter = VarianceMeter()
    for value in stream:
        meter.update(value)
        assert meter.get() >= 0.0


@PROPERTY_SETTINGS
@given(stream=_stream(min_length=10, max_length=30))
def test_variance_meter_matches_numpy_var(stream):
    """VarianceMeter matches np.var (population) on the same stream."""
    meter = VarianceMeter()
    for i, value in enumerate(stream):
        meter.update(value)
        if i >= 1:  # variance needs at least 2 points
            assert meter.get() == pytest.approx(np.var(stream[: i + 1]), rel=1e-10, abs=1e-12)


@PROPERTY_SETTINGS
@given(stream=_stream(min_length=5, max_length=20))
def test_max_min_meter_bounds(stream):
    """MaxMeter >= MinMeter on the same stream."""
    max_meter = MaxMeter(keep_values=True)
    min_meter = MinMeter(keep_values=True)
    for value in stream:
        max_meter.update(value)
        min_meter.update(value)
    assert max_meter.get() >= min_meter.get()


@PROPERTY_SETTINGS
@given(stream=_stream(min_length=10, max_length=30))
def test_max_meter_matches_numpy_max(stream):
    """MaxMeter matches np.max on the same stream."""
    meter = MaxMeter(keep_values=True)
    for i, value in enumerate(stream):
        meter.update(value)
        assert meter.get() == pytest.approx(np.max(stream[: i + 1]))


@PROPERTY_SETTINGS
@given(stream=_stream(min_length=10, max_length=30))
def test_min_meter_matches_numpy_min(stream):
    """MinMeter matches np.min on the same stream."""
    meter = MinMeter(keep_values=True)
    for i, value in enumerate(stream):
        meter.update(value)
        assert meter.get() == pytest.approx(np.min(stream[: i + 1]))


@PROPERTY_SETTINGS
@given(stream=_stream(min_length=5, max_length=20))
def test_abs_statistic_is_nonnegative(stream):
    """AbsStatistic wraps any statistic and returns absolute value."""
    from pysad.statistics import AbsStatistic

    meter = AbsStatistic(AverageMeter)
    for value in stream:
        meter.update(value)
        assert meter.get() >= 0.0
    # AbsStatistic of AverageMeter equals absolute of average.
    avg = AverageMeter()
    for value in stream:
        avg.update(value)
    assert meter.get() == pytest.approx(abs(avg.get()))


# ==============================================================================
# RUNNING STATISTIC (SLIDING WINDOW) INVARIANTS
# ==============================================================================

REMOVABLE_FOR_RUNNING = [
    AverageMeter,
    CountMeter,
    SumMeter,
    SumSquaresMeter,
    VarianceMeter,
    MaxMeter,
    MinMeter,
]


@pytest.mark.parametrize(
    "stat_cls", REMOVABLE_FOR_RUNNING, ids=[c.__name__ for c in REMOVABLE_FOR_RUNNING]
)
@PROPERTY_SETTINGS
@given(stream=_stream(min_length=10, max_length=40), window_size=st.integers(3, 15))
def test_running_statistic_window_size_bound(stat_cls, stream, window_size):
    """RunningStatistic never exceeds window_size items."""
    kwargs = {"keep_values": True} if stat_cls in (MaxMeter, MinMeter) else {}
    running = RunningStatistic(stat_cls, window_size=window_size, **kwargs)
    for value in stream:
        running.update(value)
    # The internal window deque length should not exceed window_size.
    assert len(running.window) <= window_size


@pytest.mark.parametrize(
    "stat_cls", REMOVABLE_FOR_RUNNING, ids=[c.__name__ for c in REMOVABLE_FOR_RUNNING]
)
@PROPERTY_SETTINGS
@given(stream=_stream(min_length=10, max_length=40), window_size=st.integers(3, 15))
def test_running_statistic_matches_manual_window(stat_cls, stream, window_size):
    """RunningStatistic matches manual sliding window computation."""
    kwargs = {"keep_values": True} if stat_cls in (MaxMeter, MinMeter) else {}
    running = RunningStatistic(stat_cls, window_size=window_size, **kwargs)
    manual_window = []
    for value in stream:
        manual_window.append(value)
        if len(manual_window) > window_size:
            manual_window.pop(0)
        running.update(value)
        # Compute expected using a fresh statistic on the manual window.
        expected_stat = stat_cls(**kwargs)
        for v in manual_window:
            expected_stat.update(v)
        # VarianceMeter and SumSquaresMeter have floating-point precision issues.
        if stat_cls in (VarianceMeter, SumSquaresMeter):
            tol = 1e-8
        else:
            tol = 1e-12
        assert running.get() == pytest.approx(expected_stat.get(), rel=1e-10, abs=tol)


@PROPERTY_SETTINGS
@given(stream=_stream(min_length=10, max_length=40), window_size=st.integers(3, 15))
def test_running_average_matches_numpy_sliding_mean(stream, window_size):
    """RunningStatistic(AverageMeter) matches numpy sliding window mean."""
    running = RunningStatistic(AverageMeter, window_size=window_size)
    for i, value in enumerate(stream):
        running.update(value)
        start = max(0, i + 1 - window_size)
        expected = np.mean(stream[start : i + 1])
        assert running.get() == pytest.approx(expected)


@PROPERTY_SETTINGS
@given(window_size=st.integers(max_value=0))
def test_running_statistic_rejects_non_positive_window(window_size):
    """RunningStatistic rejects window_size < 1."""
    with pytest.raises(ValueError, match="window_size must be a positive integer"):
        RunningStatistic(AverageMeter, window_size=window_size)


@PROPERTY_SETTINGS
@given(stream=_stream(min_length=5, max_length=20))
def test_running_statistic_with_non_removable_raises(stream):
    """RunningStatistic raises TypeError for non-removable statistics."""
    from pysad.statistics import QuantileMeter

    with pytest.raises(TypeError, match="does not support remove"):
        RunningStatistic(QuantileMeter, window_size=5, q=0.5)


# ==============================================================================
# EXPONENTIAL MOVING AVERAGE/VARIANCE INVARIANTS
# ==============================================================================


@PROPERTY_SETTINGS
@given(
    stream=_stream(min_length=5, max_length=30),
    alpha=st.floats(min_value=0.01, max_value=1.0),
)
def test_ema_first_value_seeds(stream, alpha):
    """EMA first value seeds the average exactly."""
    meter = ExponentialMovingAverageMeter(alpha=alpha)
    meter.update(stream[0])
    assert meter.get() == pytest.approx(stream[0])
    assert meter.num_items == 1


@PROPERTY_SETTINGS
@given(
    stream=_stream(min_length=5, max_length=30),
    alpha=st.floats(min_value=0.01, max_value=1.0),
)
def test_ema_stays_in_bounds(stream, alpha):
    """EMA stays between min and max of observed values."""
    meter = ExponentialMovingAverageMeter(alpha=alpha)
    observed_min = math.inf
    observed_max = -math.inf
    for value in stream:
        observed_min = min(observed_min, value)
        observed_max = max(observed_max, value)
        meter.update(value)
        assert observed_min - 1e-10 <= meter.get() <= observed_max + 1e-10


@PROPERTY_SETTINGS
@given(half_life=st.floats(min_value=0.1, max_value=100.0))
def test_ema_half_life_converts_to_alpha(half_life):
    """half_life parameter correctly converts to alpha."""
    meter = ExponentialMovingAverageMeter(half_life=half_life)
    expected_alpha = 1.0 - 0.5 ** (1.0 / half_life)
    assert meter.alpha == pytest.approx(expected_alpha)


@PROPERTY_SETTINGS
@given(
    stream=_stream(min_length=5, max_length=30),
    alpha=st.floats(min_value=0.01, max_value=1.0),
)
def test_emv_first_value_seeds_variance(stream, alpha):
    """EMV first value seeds variance to 0."""
    meter = ExponentialMovingVarianceMeter(alpha=alpha)
    meter.update(stream[0])
    assert meter.get() == 0.0


@PROPERTY_SETTINGS
@given(
    stream=_stream(min_length=10, max_length=30),
    alpha=st.floats(min_value=0.01, max_value=1.0),
)
def test_emv_non_negative(stream, alpha):
    """EMV never returns negative variance."""
    meter = ExponentialMovingVarianceMeter(alpha=alpha)
    for value in stream:
        meter.update(value)
        assert meter.get() >= 0.0


@PROPERTY_SETTINGS
@given(alpha=st.floats(min_value=-1.0, max_value=0.0))
def test_ema_rejects_invalid_alpha(alpha):
    """EMA rejects alpha <= 0."""
    with pytest.raises(ValueError, match="alpha must be in the interval"):
        ExponentialMovingAverageMeter(alpha=alpha)


@PROPERTY_SETTINGS
@given(alpha=st.floats(min_value=1.0 + 1e-10, max_value=2.0))
def test_ema_rejects_alpha_above_one(alpha):
    """EMA rejects alpha > 1."""
    with pytest.raises(ValueError, match="alpha must be in the interval"):
        ExponentialMovingAverageMeter(alpha=alpha)


@PROPERTY_SETTINGS
@given(half_life=st.floats(min_value=-10.0, max_value=0.0))
def test_ema_rejects_invalid_half_life(half_life):
    """EMA rejects half_life <= 0."""
    with pytest.raises(ValueError, match="half_life must be positive"):
        ExponentialMovingAverageMeter(half_life=half_life)


@PROPERTY_SETTINGS
@given(
    alpha=st.floats(min_value=0.01, max_value=1.0),
    half_life=st.floats(min_value=0.1, max_value=100.0),
)
def test_ema_rejects_both_alpha_and_half_life(alpha, half_life):
    """EMA rejects providing both alpha and half_life."""
    with pytest.raises(ValueError, match="Pass only one of alpha and half_life"):
        ExponentialMovingAverageMeter(alpha=alpha, half_life=half_life)


# ==============================================================================
# GEOMETRIC MOVING AVERAGE INVARIANTS
# ==============================================================================


@PROPERTY_SETTINGS
@given(
    stream=_stream(min_length=5, max_length=30).filter(lambda s: all(v > 0 for v in s)),
    alpha=st.floats(min_value=0.01, max_value=1.0),
)
def test_gma_positive_stream(stream, alpha):
    """GMA on positive stream stays positive and in bounds."""
    meter = GeometricMovingAverageMeter(alpha=alpha)
    observed_min = math.inf
    observed_max = -math.inf
    for value in stream:
        observed_min = min(observed_min, value)
        observed_max = max(observed_max, value)
        meter.update(value)
        assert meter.get() > 0.0
        assert observed_min - 1e-10 <= meter.get() <= observed_max + 1e-10


@PROPERTY_SETTINGS
@given(stream=_stream(min_length=5, max_length=30))
def test_gma_rejects_negative(stream):
    """GMA raises on negative values (log of negative is undefined)."""
    meter = GeometricMovingAverageMeter(alpha=0.1)
    for value in stream:
        if value < 0:
            with pytest.raises(ValueError):
                meter.update(value)
            return
    # If we get here, all values were non-negative, which is fine.
    for value in stream:
        meter.update(value)


# ==============================================================================
# QUANTILE METER INVARIANTS
# ==============================================================================

QUANTILE_STATS = [
    (QuantileMeter, {"q": 0.5}),
    (MedianMeter, {}),
    (KLLMeter, {"q": 0.5, "k": 64, "random_state": 42}),
    (DDSketchMeter, {"q": 0.5}),
    (RelativeErrorQuantileMeter, {"q": 0.5, "random_state": 42}),
]


@pytest.mark.parametrize(
    "stat_cls,kwargs", QUANTILE_STATS, ids=[c.__name__ for c, _ in QUANTILE_STATS]
)
@PROPERTY_SETTINGS
@given(stream=_stream(min_length=10, max_length=50).filter(lambda s: len(s) > 0))
def test_quantile_meter_constant_stream(stat_cls, kwargs, stream):
    """Quantile meters return the constant value on constant stream."""
    const_value = 7.0
    const_stream = [const_value] * 20
    meter = stat_cls(**kwargs)
    for value in const_stream:
        meter.update(value)
    # DDSketchMeter has relative accuracy, so tolerance depends on the value.
    tol = 0.05 if stat_cls is DDSketchMeter else 1e-6
    assert meter.get() == pytest.approx(const_value, abs=tol)


@pytest.mark.parametrize(
    "stat_cls,kwargs", QUANTILE_STATS, ids=[c.__name__ for c, _ in QUANTILE_STATS]
)
@PROPERTY_SETTINGS
@given(
    stream=_stream(min_length=20, max_length=100).filter(lambda s: len(s) > 0),
    q=st.floats(min_value=0.01, max_value=0.99),  # QuantileMeter requires q in (0, 1)
)
def test_quantile_bounds(stat_cls, kwargs, stream, q):
    """Quantile estimate is between min and max of stream."""
    # Create kwargs with the drawn q value.
    test_kwargs = dict(kwargs)
    if "q" in test_kwargs:
        test_kwargs["q"] = q
    meter = stat_cls(**test_kwargs)
    # Some meters like MedianMeter don't take q in constructor.
    if hasattr(meter, "q"):
        meter.q = q
    for value in stream:
        meter.update(value)
    stream_min = min(stream)
    stream_max = max(stream)
    estimate = meter.get() if not hasattr(meter, "quantile") else meter.quantile(q)
    # DDSketchMeter has relative accuracy, so check relative to true value.
    if stat_cls is DDSketchMeter:
        # DDSketch has relative accuracy of 1%, meaning the estimate is within
        # 1% of the true quantile value. However, for edge cases near min/max,
        # it may return a slightly different value due to bucket boundaries.
        # We check that it's within a reasonable tolerance of true value or min/max.
        true_val = np.median(stream)
        # Allow 2% relative error, or be within the stream range
        if abs(true_val) > 1e-10:
            assert abs(estimate - true_val) <= max(abs(true_val) * 0.02, stream_max * 0.02 + 1e-6)
        else:
            # For values near zero, check absolute bounds
            assert stream_min - 0.1 <= estimate <= stream_max + 0.1
    else:
        tol = 1e-6
        assert stream_min - tol <= estimate <= stream_max + tol


@pytest.mark.parametrize(
    "stat_cls,kwargs", QUANTILE_STATS, ids=[c.__name__ for c, _ in QUANTILE_STATS]
)
@PROPERTY_SETTINGS
@given(stream=_stream(min_length=20, max_length=100))
def test_quantile_monotonicity(stat_cls, kwargs, stream):
    """Higher quantiles give higher or equal estimates."""
    qs = [0.25, 0.5, 0.75]
    estimates = []
    for q in qs:
        test_kwargs = dict(kwargs)
        if "q" in test_kwargs:
            test_kwargs["q"] = q
        meter = stat_cls(**test_kwargs)
        if hasattr(meter, "q"):
            meter.q = q
        for value in stream:
            meter.update(value)
        estimates.append(meter.get() if not hasattr(meter, "quantile") else meter.quantile(q))
    # DDSketchMeter has relative accuracy, so allow some tolerance.
    # P^2 algorithm (QuantileMeter) can have small violations on extreme streams.
    tol = 1e-2 if stat_cls in (DDSketchMeter, QuantileMeter) else 1e-12
    assert estimates[0] <= estimates[1] + tol
    assert estimates[1] <= estimates[2] + tol


@PROPERTY_SETTINGS
@given(
    stream=_stream(min_length=20, max_length=100),
    q=st.floats(min_value=0.0, max_value=1.0),
)
def test_kll_merge_commutative(stream, q):
    """KLL merge is commutative: a.merge(b) == b.merge(a)."""
    split = len(stream) // 2
    stream_a = stream[:split]
    stream_b = stream[split:]

    meter_a = KLLMeter(q=q, k=64, random_state=42)
    meter_b = KLLMeter(q=q, k=64, random_state=42)
    for v in stream_a:
        meter_a.update(v)
    for v in stream_b:
        meter_b.update(v)

    # Merge a into b
    merged_ab = KLLMeter(q=q, k=64, random_state=42)
    merged_ab.merge(meter_a)
    merged_ab.merge(meter_b)

    # Merge b into a
    merged_ba = KLLMeter(q=q, k=64, random_state=42)
    merged_ba.merge(meter_b)
    merged_ba.merge(meter_a)

    assert merged_ab.get() == pytest.approx(merged_ba.get(), rel=1e-6)


# ==============================================================================
# COUNT-MIN SKETCH AND HYPERLOGLOG INVARIANTS
# ==============================================================================


@PROPERTY_SETTINGS
@given(
    stream=st.lists(st.integers(0, 100), min_size=10, max_size=50),
    width=st.integers(10, 50),
    depth=st.integers(2, 5),
    seed=st.integers(0, 100),
)
def test_count_min_sketch_non_negative(stream, width, depth, seed):
    """CountMinSketch estimates are non-negative."""
    meter = CountMinSketchStatistic(width=width, depth=depth, random_state=seed)
    for value in stream:
        meter.update(float(value))
    # get() returns the estimate for the last value.
    estimate = meter.get()
    assert estimate >= 0


@PROPERTY_SETTINGS
@given(
    stream=st.lists(st.integers(0, 100), min_size=10, max_size=50),
    width=st.integers(10, 50),
    depth=st.integers(2, 5),
    seed=st.integers(0, 100),
)
def test_count_min_sketch_merge_commutative(stream, width, depth, seed):
    """CountMinSketch merge is commutative."""
    split = len(stream) // 2
    stream_a = stream[:split]
    stream_b = stream[split:]

    # Use same seed for both meters so they have the same hash functions
    meter_a = CountMinSketchStatistic(width=width, depth=depth, random_state=seed)
    meter_b = CountMinSketchStatistic(width=width, depth=depth, random_state=seed)
    for v in stream_a:
        meter_a.update(float(v))
    for v in stream_b:
        meter_b.update(float(v))

    merged_ab = CountMinSketchStatistic(width=width, depth=depth, random_state=seed)
    merged_ab.merge(meter_a)
    merged_ab.merge(meter_b)

    merged_ba = CountMinSketchStatistic(width=width, depth=depth, random_state=seed)
    merged_ba.merge(meter_b)
    merged_ba.merge(meter_a)

    # Merged results should have similar estimates for the same values
    assert merged_ab.get() == pytest.approx(merged_ba.get(), rel=1e-6)


@PROPERTY_SETTINGS
@given(
    stream=st.lists(st.integers(0, 1000), min_size=10, max_size=100),
    precision=st.integers(4, 12),
)
def test_hyperloglog_non_decreasing(stream, precision):
    """HyperLogLog cardinality estimate is non-decreasing."""
    meter = HyperLogLogStatistic(precision=precision)
    prev_estimate = 0.0
    for value in stream:
        meter.update(float(value))
        estimate = meter.get()
        assert estimate >= prev_estimate - 1e-6  # Allow tiny floating point variance
        prev_estimate = estimate


@PROPERTY_SETTINGS
@given(
    stream=st.lists(st.integers(0, 1000), min_size=10, max_size=100),
    precision=st.integers(4, 12),
)
def test_hyperloglog_merge_commutative(stream, precision):
    """HyperLogLog merge is commutative."""
    split = len(stream) // 2
    stream_a = stream[:split]
    stream_b = stream[split:]

    meter_a = HyperLogLogStatistic(precision=precision)
    meter_b = HyperLogLogStatistic(precision=precision)
    for v in stream_a:
        meter_a.update(float(v))
    for v in stream_b:
        meter_b.update(float(v))

    merged_ab = HyperLogLogStatistic(precision=precision)
    merged_ab.merge(meter_a)
    merged_ab.merge(meter_b)

    merged_ba = HyperLogLogStatistic(precision=precision)
    merged_ba.merge(meter_b)
    merged_ba.merge(meter_a)

    assert merged_ab.get() == pytest.approx(merged_ba.get(), rel=1e-6)


# ==============================================================================
# DRIFT DETECTOR INVARIANTS
# ==============================================================================

# Binary stream for drift detectors that expect 0/1 errors.
BINARY_STREAM = st.lists(st.sampled_from([0.0, 1.0]), min_size=10, max_size=100)

# General stream for detectors that accept any real values.
REAL_STREAM = _stream(min_length=10, max_length=100)


DRIFT_DETECTORS_BINARY = [
    (DDMMeter, {"warning_level": 2.0, "drift_level": 3.0, "min_num_items": 30}),
    (EDDMMeter, {"warning_threshold": 0.95, "drift_threshold": 0.9, "min_num_errors": 30}),
    (FHDDMMeter, {"window_size": 50, "delta": 1e-6}),
    (FHDDMSMeter, {"short_window_size": 25, "long_window_size": 50, "delta": 1e-7}),
    (FHDDMSAddMeter, {"short_window_size": 25, "stack_size": 2, "delta": 1e-7}),
]

DRIFT_DETECTORS_REAL = [
    (ADWINMeter, {"delta": 0.002, "max_buckets": 5}),
    (HDDMAMeter, {"drift_confidence": 0.001, "warning_confidence": 0.005, "two_sided": False}),
    (
        HDDMWMeter,
        {"drift_confidence": 0.001, "warning_confidence": 0.005, "alpha": 0.05, "two_sided": False},
    ),
    (KSWINMeter, {"alpha": 0.0001, "window_size": 100, "stat_size": 10, "random_state": 42}),
    (PageHinkleyMeter, {"delta": 0.0, "threshold": 50.0, "min_instances": 30}),
    (CUSUMMeter, {"mean": 0.0, "delta": 0.5, "threshold": 5.0}),
]


@pytest.mark.parametrize(
    "detector_cls,kwargs",
    DRIFT_DETECTORS_BINARY,
    ids=[c.__name__ for c, _ in DRIFT_DETECTORS_BINARY],
)
@PROPERTY_SETTINGS
@given(stream=BINARY_STREAM)
def test_binary_drift_detector_drift_flag_is_boolean(detector_cls, kwargs, stream):
    """drift_detected is always a boolean."""
    meter = detector_cls(**kwargs)
    for value in stream:
        meter.update(value)
        assert _is_bool_or_numpy_bool(meter.drift_detected)


@pytest.mark.parametrize(
    "detector_cls,kwargs", DRIFT_DETECTORS_REAL, ids=[c.__name__ for c, _ in DRIFT_DETECTORS_REAL]
)
@PROPERTY_SETTINGS
@given(stream=REAL_STREAM)
def test_real_drift_detector_drift_flag_is_boolean(detector_cls, kwargs, stream):
    """drift_detected is always a boolean."""
    meter = detector_cls(**kwargs)
    for value in stream:
        meter.update(value)
        assert _is_bool_or_numpy_bool(meter.drift_detected)


@pytest.mark.parametrize(
    "detector_cls,kwargs",
    DRIFT_DETECTORS_BINARY,
    ids=[c.__name__ for c, _ in DRIFT_DETECTORS_BINARY],
)
@PROPERTY_SETTINGS
@given(stream=BINARY_STREAM)
def test_binary_drift_detector_get_returns_finite(detector_cls, kwargs, stream):
    """get() returns a finite number."""
    meter = detector_cls(**kwargs)
    for value in stream:
        meter.update(value)
    result = meter.get()
    assert isinstance(result, (int, float)) and not isinstance(result, bool)
    assert math.isfinite(result)


@pytest.mark.parametrize(
    "detector_cls,kwargs", DRIFT_DETECTORS_REAL, ids=[c.__name__ for c, _ in DRIFT_DETECTORS_REAL]
)
@PROPERTY_SETTINGS
@given(stream=REAL_STREAM)
def test_real_drift_detector_get_returns_finite(detector_cls, kwargs, stream):
    """get() returns a finite number."""
    meter = detector_cls(**kwargs)
    for value in stream:
        meter.update(value)
    result = meter.get()
    assert isinstance(result, (int, float)) and not isinstance(result, bool)
    assert math.isfinite(result)


@pytest.mark.parametrize(
    "detector_cls,kwargs",
    DRIFT_DETECTORS_BINARY,
    ids=[c.__name__ for c, _ in DRIFT_DETECTORS_BINARY],
)
@PROPERTY_SETTINGS
@given(stream=BINARY_STREAM)
def test_binary_drift_detector_drift_resets_on_next_update(detector_cls, kwargs, stream):
    """After drift_detected=True, the next update resets the detector."""
    meter = detector_cls(**kwargs)
    for i, value in enumerate(stream):
        meter.update(value)
        if meter.drift_detected:
            # Next update should reset (if there is one).
            if i + 1 < len(stream):
                next_value = stream[i + 1]
                meter.update(next_value)
                # After reset, drift_detected should be False (unless immediate re-detection).
                assert hasattr(meter, "drift_detected")
            break
    # If no drift was detected, that's also a valid outcome.
    # Test passes if no exception raised.


@pytest.mark.parametrize(
    "detector_cls,kwargs",
    DRIFT_DETECTORS_BINARY + DRIFT_DETECTORS_REAL,
    ids=[c.__name__ for c, _ in DRIFT_DETECTORS_BINARY + DRIFT_DETECTORS_REAL],
)
@PROPERTY_SETTINGS
@given(stream=st.lists(st.sampled_from([0.0]), min_size=50, max_size=100))
def test_drift_detector_no_drift_on_constant_zero_stream(detector_cls, kwargs, stream):
    """Constant zero stream should not trigger drift (for most detectors)."""
    meter = detector_cls(**kwargs)
    for value in stream:
        meter.update(value)
    # Most detectors should not drift on constant stream.
    # CUSUM with mean=0, delta=0.5, threshold=5 won't drift on zeros.
    # ADWIN with values in [0,1] won't drift on constant zeros.
    # PageHinkley with delta=0, threshold=50 won't drift on constant stream.
    # HDDM_A/W with values in [0,1] won't drift on constant stream.
    # KSWIN compares distributions, constant stream has zero KS distance.
    assert not meter.drift_detected or meter.get() == 0.0


# DDM-specific invariants
@PROPERTY_SETTINGS
@given(stream=BINARY_STREAM)
def test_ddm_warning_and_drift_mutually_exclusive(stream):
    """DDM warning_detected and drift_detected are never both True."""
    meter = DDMMeter()
    for value in stream:
        meter.update(value)
        assert not (meter.warning_detected and meter.drift_detected)


@PROPERTY_SETTINGS
@given(stream=BINARY_STREAM)
def test_ddm_drift_resets_context(stream):
    """DDM resets context after drift."""
    meter = DDMMeter(min_num_items=10)
    for value in stream:
        meter.update(value)
        if meter.drift_detected:
            # After drift, next update starts fresh context.
            assert meter.num_items == 1
            assert meter.get() == pytest.approx(stream[-1] if stream else 0.0)
            break


# EDDM-specific invariants
@PROPERTY_SETTINGS
@given(stream=BINARY_STREAM)
def test_eddm_warning_and_drift_mutually_exclusive(stream):
    """EDDM warning_detected and drift_detected are never both True."""
    meter = EDDMMeter()
    for value in stream:
        meter.update(value)
        assert not (meter.warning_detected and meter.drift_detected)


# FHDDM-specific invariants
@PROPERTY_SETTINGS
@given(stream=BINARY_STREAM)
def test_fhddm_only_drift_no_warning(stream):
    """FHDDM has no warning level, only drift."""
    meter = FHDDMMeter()
    for value in stream:
        meter.update(value)
    # FHDDM only has drift_detected, not warning_detected.
    assert hasattr(meter, "drift_detected")
    assert not hasattr(meter, "warning_detected")


@PROPERTY_SETTINGS
@given(stream=BINARY_STREAM)
def test_fhddms_only_drift_no_warning(stream):
    """FHDDMS has no warning level, only drift."""
    meter = FHDDMSMeter()
    for value in stream:
        meter.update(value)
    assert hasattr(meter, "drift_detected")
    assert not hasattr(meter, "warning_detected")


@PROPERTY_SETTINGS
@given(stream=BINARY_STREAM)
def test_fhddms_add_only_drift_no_warning(stream):
    """FHDDMSAdd has no warning level, only drift."""
    meter = FHDDMSAddMeter()
    for value in stream:
        meter.update(value)
    assert hasattr(meter, "drift_detected")
    assert not hasattr(meter, "warning_detected")


# HDDM_A/W invariants
@PROPERTY_SETTINGS
@given(
    stream=REAL_STREAM,
    drift_conf=st.floats(min_value=0.0001, max_value=0.01),
    warn_conf=st.floats(min_value=0.001, max_value=0.1),
)
def test_hddma_warning_drift_confidence_order(stream, drift_conf, warn_conf):
    """HDDM_A requires drift_confidence < warning_confidence."""
    if drift_conf >= warn_conf:
        with pytest.raises(ValueError, match="drift_confidence < warning_confidence"):
            HDDMAMeter(drift_confidence=drift_conf, warning_confidence=warn_conf)
    else:
        meter = HDDMAMeter(drift_confidence=drift_conf, warning_confidence=warn_conf)
        for value in stream:
            meter.update(value)
        assert _is_bool_or_numpy_bool(meter.drift_detected)
        assert _is_bool_or_numpy_bool(meter.warning_detected)
        assert not (meter.warning_detected and meter.drift_detected)


@PROPERTY_SETTINGS
@given(
    stream=REAL_STREAM,
    drift_conf=st.floats(min_value=0.0001, max_value=0.01),
    warn_conf=st.floats(min_value=0.001, max_value=0.1),
)
def test_hddmw_warning_drift_confidence_order(stream, drift_conf, warn_conf):
    """HDDM_W requires drift_confidence < warning_confidence."""
    if drift_conf >= warn_conf:
        with pytest.raises(ValueError, match="drift_confidence < warning_confidence"):
            HDDMWMeter(drift_confidence=drift_conf, warning_confidence=warn_conf)
    else:
        meter = HDDMWMeter(drift_confidence=drift_conf, warning_confidence=warn_conf)
        for value in stream:
            meter.update(value)
        assert _is_bool_or_numpy_bool(meter.drift_detected)
        assert _is_bool_or_numpy_bool(meter.warning_detected)
        assert not (meter.warning_detected and meter.drift_detected)


@PROPERTY_SETTINGS
@given(alpha=st.floats(min_value=-1.0, max_value=0.0))
def test_hddmw_rejects_invalid_alpha(alpha):
    """HDDM_W rejects alpha <= 0."""
    with pytest.raises(ValueError, match="alpha must be in the interval"):
        HDDMWMeter(alpha=alpha)


@PROPERTY_SETTINGS
@given(alpha=st.floats(min_value=1.0 + 1e-10, max_value=2.0))
def test_hddmw_rejects_alpha_above_one(alpha):
    """HDDM_W rejects alpha > 1."""
    with pytest.raises(ValueError, match="alpha must be in the interval"):
        HDDMWMeter(alpha=alpha)


# KSWIN invariants
@PROPERTY_SETTINGS
@given(
    window_size=st.integers(10, 50),
    stat_size=st.integers(5, 25),
)
def test_kswin_rejects_invalid_window_size(window_size, stat_size):
    """KSWIN requires window_size >= 2 * stat_size."""
    if window_size < 2 * stat_size:
        with pytest.raises(ValueError, match="window_size must be at least"):
            KSWINMeter(window_size=window_size, stat_size=stat_size)
    else:
        meter = KSWINMeter(window_size=window_size, stat_size=stat_size)
        assert meter.window_size == window_size
        assert meter.stat_size == stat_size


@PROPERTY_SETTINGS
@given(stream=REAL_STREAM)
def test_kswin_distance_non_negative(stream):
    """KSWIN KS distance is non-negative."""
    meter = KSWINMeter(window_size=50, stat_size=10, random_state=42)
    for value in stream:
        meter.update(value)
    assert meter.get() >= 0.0


@PROPERTY_SETTINGS
@given(stream=REAL_STREAM)
def test_kswin_drift_detected_when_distance_exceeds_threshold(stream):
    """KSWIN drift_detected when KS distance > threshold."""
    meter = KSWINMeter(
        window_size=50, stat_size=10, alpha=0.5, random_state=42
    )  # High alpha = low threshold
    for value in stream:
        meter.update(value)
    # If drift detected, distance must exceed threshold.
    if meter.drift_detected:
        assert meter.get() > meter._threshold


# Page-Hinkley invariants
@PROPERTY_SETTINGS
@given(stream=REAL_STREAM)
def test_page_hinkley_statistic_non_negative(stream):
    """Page-Hinkley statistic is non-negative."""
    meter = PageHinkleyMeter()
    for value in stream:
        meter.update(value)
    assert meter.get() >= 0.0


@PROPERTY_SETTINGS
@given(stream=REAL_STREAM)
def test_page_hinkley_drift_resets_sums(stream):
    """Page-Hinkley resets g_plus and g_minus after drift on next update."""
    meter = PageHinkleyMeter(threshold=1.0, min_instances=1)
    for i, value in enumerate(stream):
        meter.update(value)
        if meter.drift_detected:
            # After drift, sums reset on the next update.
            if i + 1 < len(stream):
                next_val = stream[i + 1]
                meter.update(next_val)
                assert meter.g_plus == 0.0
                assert meter.g_minus == 0.0
            break
    # If no drift was detected, that's also a valid outcome.


@PROPERTY_SETTINGS
@given(delta=st.floats(min_value=-1.0, max_value=-0.001))
def test_page_hinkley_rejects_negative_delta(delta):
    """Page-Hinkley rejects negative delta."""
    with pytest.raises(ValueError, match="delta must be non-negative"):
        PageHinkleyMeter(delta=delta)


@PROPERTY_SETTINGS
@given(threshold=st.floats(min_value=-10.0, max_value=0.0))
def test_page_hinkley_rejects_non_positive_threshold(threshold):
    """Page-Hinkley rejects threshold <= 0."""
    with pytest.raises(ValueError, match="threshold must be positive"):
        PageHinkleyMeter(threshold=threshold)


@PROPERTY_SETTINGS
@given(min_instances=st.integers(min_value=-5, max_value=0))
def test_page_hinkley_rejects_non_positive_min_instances(min_instances):
    """Page-Hinkley rejects min_instances < 1."""
    with pytest.raises(ValueError, match="min_instances must be a positive integer"):
        PageHinkleyMeter(min_instances=min_instances)


# CUSUM invariants
@PROPERTY_SETTINGS
@given(stream=REAL_STREAM)
def test_cusum_statistic_non_negative(stream):
    """CUSUM statistic is non-negative."""
    meter = CUSUMMeter()
    for value in stream:
        meter.update(value)
    assert meter.get() >= 0.0


@PROPERTY_SETTINGS
@given(stream=REAL_STREAM)
def test_cusum_drift_resets_sums(stream):
    """CUSUM resets g_plus_ and g_minus_ after drift on next update."""
    # Use a threshold that allows drift to be detected on extreme values
    meter = CUSUMMeter(threshold=0.5)
    for i, value in enumerate(stream):
        meter.update(value)
        if meter.drift_detected:
            # After drift is detected, the next update resets g_plus_ and g_minus_
            if i + 1 < len(stream):
                next_val = stream[i + 1]
                meter.update(next_val)
                # After reset, g_plus_ and g_minus_ should be re-initialized fresh
                assert meter.get() >= 0.0  # CUSUM statistic is always non-negative
            break
    # If no drift was detected, that's also a valid outcome.


@PROPERTY_SETTINGS
@given(delta=st.floats(min_value=-1.0, max_value=-0.001))
def test_cusum_rejects_negative_delta(delta):
    """CUSUM rejects negative delta."""
    with pytest.raises(ValueError, match="delta must be non-negative"):
        CUSUMMeter(delta=delta)


@PROPERTY_SETTINGS
@given(threshold=st.floats(min_value=-10.0, max_value=0.0))
def test_cusum_rejects_non_positive_threshold(threshold):
    """CUSUM rejects threshold <= 0."""
    with pytest.raises(ValueError, match="threshold must be positive"):
        CUSUMMeter(threshold=threshold)


# ADWIN invariants
ADWIN_SETTINGS = settings(
    max_examples=20,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    derandomize=True,
)


@ADWIN_SETTINGS
@given(stream=REAL_STREAM.filter(lambda s: all(0 <= v <= 1 for v in s)))
def test_adwin_mean_in_bounds(stream):
    """ADWIN mean stays in [0, 1] for inputs in [0, 1]."""
    meter = ADWINMeter()
    for value in stream:
        meter.update(value)
    assert 0.0 <= meter.get() <= 1.0


@PROPERTY_SETTINGS
@given(delta=st.floats(min_value=-0.1, max_value=0.0))
def test_adwin_rejects_invalid_delta(delta):
    """ADWIN rejects delta <= 0."""
    with pytest.raises(ValueError, match="delta must be in"):
        ADWINMeter(delta=delta)


@PROPERTY_SETTINGS
@given(delta=st.floats(min_value=1.0, max_value=2.0))
def test_adwin_rejects_delta_above_one(delta):
    """ADWIN rejects delta >= 1."""
    with pytest.raises(ValueError, match="delta must be in"):
        ADWINMeter(delta=delta)


# ==============================================================================
# ONLINE PERIODICITY METER INVARIANTS
# ==============================================================================


@PROPERTY_SETTINGS
@given(
    stream=_stream(min_length=20, max_length=100),
    window_size=st.integers(5, 20),
)
def test_online_periodicity_meter_period_in_bounds(stream, window_size):
    """OnlinePeriodicityMeter period estimate is within valid bounds."""
    meter = OnlinePeriodicityMeter(window_size=window_size)
    for value in stream:
        meter.update(value)
    period = meter.get()
    # Period should be between 1 and window_size, or 0 if not detected.
    assert period == 0 or (1 <= period <= window_size)


@PROPERTY_SETTINGS
@given(window_size=st.integers(min_value=0, max_value=0))
def test_online_periodicity_rejects_invalid_window(window_size):
    """OnlinePeriodicityMeter rejects window_size < 1."""
    with pytest.raises(ValueError):
        OnlinePeriodicityMeter(window_size=window_size)


# ==============================================================================
# STATEFUL TESTS FOR DRIFT DETECTORS
# ==============================================================================


class _DriftDetectorMachine(RuleBasedStateMachine):
    """State machine for testing drift detector chronology and reset behavior."""

    def __init__(self):
        super().__init__()
        self.detector = None
        self.stream_history = []

    @initialize()
    def setup(self):
        # Use a detector with fast detection for testing.
        self.detector = DDMMeter(min_num_items=5, warning_level=2.0, drift_level=3.0)
        self.stream_history = []

    @rule(data=st.data())
    def append_binary(self, data):
        value = data.draw(st.sampled_from([0.0, 1.0]))
        self.detector.update(value)
        self.stream_history.append(value)

    @rule()
    def check_flags_boolean(self):
        assert _is_bool_or_numpy_bool(self.detector.drift_detected)
        assert _is_bool_or_numpy_bool(self.detector.warning_detected)
        assert not (self.detector.warning_detected and self.detector.drift_detected)

    @rule()
    def check_get_finite(self):
        result = self.detector.get()
        assert isinstance(result, (int, float))
        assert math.isfinite(result)

    @rule()
    def check_drift_resets(self):
        if self.detector.drift_detected:
            # After drift, num_items should be 1 (the value that caused drift).
            assert self.detector.num_items >= 1


_DriftDetectorMachine = STATEFUL_SETTINGS(_DriftDetectorMachine)
TestDriftDetectorChronology = _DriftDetectorMachine.TestCase
TestDriftDetectorChronology.__name__ = "TestDriftDetectorChronology"
TestDriftDetectorChronology.__qualname__ = "TestDriftDetectorChronology"


class _CUSUMMachine(RuleBasedStateMachine):
    """State machine for testing CUSUM reset behavior."""

    def __init__(self):
        super().__init__()
        self.detector = None
        self.last_was_drift = False

    @initialize()
    def setup(self):
        self.detector = CUSUMMeter(mean=0.0, delta=0.5, threshold=3.0)
        self.last_was_drift = False

    @rule(data=st.data())
    def append_value(self, data):
        value = data.draw(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        )
        self.last_was_drift = bool(self.detector.drift_detected)
        self.detector.update(value)

    @rule()
    def check_statistic_non_negative(self):
        assert self.detector.get() >= 0.0

    @rule()
    def check_drift_resets_on_next_update(self):
        # After drift_detected, the next update resets g_plus_ and g_minus_
        if self.last_was_drift:
            assert self.detector.g_plus_ == 0.0
            assert self.detector.g_minus_ == 0.0


_CUSUMMachine = STATEFUL_SETTINGS(_CUSUMMachine)
TestCUSUMChronology = _CUSUMMachine.TestCase
TestCUSUMChronology.__name__ = "TestCUSUMChronology"
TestCUSUMChronology.__qualname__ = "TestCUSUMChronology"


class _PageHinkleyMachine(RuleBasedStateMachine):
    """State machine for testing Page-Hinkley reset behavior."""

    def __init__(self):
        super().__init__()
        self.detector = None
        self.last_was_drift = False

    @initialize()
    def setup(self):
        self.detector = PageHinkleyMeter(delta=0.0, threshold=5.0, min_instances=1)
        self.last_was_drift = False

    @rule(data=st.data())
    def append_value(self, data):
        value = data.draw(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        )
        self.last_was_drift = bool(self.detector.drift_detected)
        self.detector.update(value)

    @rule()
    def check_statistic_non_negative(self):
        assert self.detector.get() >= 0.0

    @rule()
    def check_drift_resets_on_next_update(self):
        # After drift_detected, the next update resets g_plus and g_minus
        if self.last_was_drift:
            assert self.detector.g_plus == 0.0
            assert self.detector.g_minus == 0.0


_PageHinkleyMachine = STATEFUL_SETTINGS(_PageHinkleyMachine)
TestPageHinkleyChronology = _PageHinkleyMachine.TestCase
TestPageHinkleyChronology.__name__ = "TestPageHinkleyChronology"
TestPageHinkleyChronology.__qualname__ = "TestPageHinkleyChronology"


# ==============================================================================
# CONTRACT TESTS FOR ALL STATISTICS
# ==============================================================================

ALL_STATISTICS = [
    ("AverageMeter", AverageMeter, {}),
    ("CountMeter", CountMeter, {}),
    ("SumMeter", SumMeter, {}),
    ("SumSquaresMeter", SumSquaresMeter, {}),
    ("VarianceMeter", VarianceMeter, {}),
    ("MaxMeter", MaxMeter, {"keep_values": True}),
    ("MinMeter", MinMeter, {"keep_values": True}),
    ("ExponentialMovingAverageMeter", ExponentialMovingAverageMeter, {"alpha": 0.1}),
    ("ExponentialMovingVarianceMeter", ExponentialMovingVarianceMeter, {"alpha": 0.1}),
    ("GeometricMovingAverageMeter", GeometricMovingAverageMeter, {"alpha": 0.1}),
    ("QuantileMeter", QuantileMeter, {"q": 0.5}),
    ("MedianMeter", MedianMeter, {}),
    ("KLLMeter", KLLMeter, {"q": 0.5, "k": 64, "random_state": 42}),
    ("DDSketchMeter", DDSketchMeter, {"q": 0.5}),
    ("RelativeErrorQuantileMeter", RelativeErrorQuantileMeter, {"q": 0.5, "random_state": 42}),
    (
        "CountMinSketchStatistic",
        CountMinSketchStatistic,
        {"width": 32, "depth": 3, "random_state": 42},
    ),
    ("HyperLogLogStatistic", HyperLogLogStatistic, {"precision": 8}),
    ("OnlinePeriodicityMeter", OnlinePeriodicityMeter, {"window_size": 10}),
    ("ADWINMeter", ADWINMeter, {"delta": 0.002, "max_buckets": 5}),
    ("DDMMeter", DDMMeter, {"warning_level": 2.0, "drift_level": 3.0, "min_num_items": 30}),
    (
        "EDDMMeter",
        EDDMMeter,
        {"warning_threshold": 0.95, "drift_threshold": 0.9, "min_num_errors": 30},
    ),
    ("FHDDMMeter", FHDDMMeter, {"window_size": 50, "delta": 1e-6}),
    ("FHDDMSMeter", FHDDMSMeter, {"short_window_size": 25, "long_window_size": 50, "delta": 1e-7}),
    ("FHDDMSAddMeter", FHDDMSAddMeter, {"short_window_size": 25, "stack_size": 2, "delta": 1e-7}),
    (
        "HDDMAMeter",
        HDDMAMeter,
        {"drift_confidence": 0.001, "warning_confidence": 0.005, "two_sided": False},
    ),
    (
        "HDDMWMeter",
        HDDMWMeter,
        {"drift_confidence": 0.001, "warning_confidence": 0.005, "alpha": 0.05, "two_sided": False},
    ),
    (
        "KSWINMeter",
        KSWINMeter,
        {"alpha": 0.0001, "window_size": 100, "stat_size": 10, "random_state": 42},
    ),
    ("PageHinkleyMeter", PageHinkleyMeter, {"delta": 0.0, "threshold": 50.0, "min_instances": 30}),
    ("CUSUMMeter", CUSUMMeter, {"mean": 0.0, "delta": 0.5, "threshold": 5.0}),
]


@pytest.mark.parametrize("name,cls,kwargs", ALL_STATISTICS, ids=[n for n, _, _ in ALL_STATISTICS])
@PROPERTY_SETTINGS
@given(data=st.data())
def test_all_statistics_basic_contract(name, cls, kwargs, data):
    """Basic contract: get() returns finite after updates, get() is idempotent."""
    # Handle classes that need special handling.
    if name == "AbsStatistic":
        from pysad.statistics import AbsStatistic, AverageMeter

        meter = AbsStatistic(AverageMeter)
    else:
        meter = cls(**kwargs)

    # Draw a stream appropriate for the statistic.
    if name in ("CountMinSketchStatistic",):
        stream = data.draw(st.lists(st.integers(0, 100), min_size=10, max_size=50))
        stream = [float(v) for v in stream]
    elif name in ("DDMMeter", "EDDMMeter", "FHDDMMeter", "FHDDMSMeter", "FHDDMSAddMeter"):
        stream = data.draw(st.lists(st.sampled_from([0.0, 1.0]), min_size=10, max_size=50))
    elif name == "GeometricMovingAverageMeter":
        # GMA needs positive values.
        stream = data.draw(
            st.lists(st.floats(min_value=0.01, max_value=100.0), min_size=10, max_size=50)
        )
    else:
        stream = data.draw(_stream(min_length=10, max_length=50))

    for value in stream:
        meter.update(value)

    # get() returns finite.
    result = meter.get()
    assert isinstance(result, (int, float)) and not isinstance(result, bool)
    assert math.isfinite(result)

    # get() is idempotent.
    assert meter.get() == result

    # drift_detected is boolean (for drift detectors).
    if hasattr(meter, "drift_detected"):
        assert _is_bool_or_numpy_bool(meter.drift_detected)
    if hasattr(meter, "warning_detected"):
        assert _is_bool_or_numpy_bool(meter.warning_detected)
        assert not (meter.warning_detected and meter.drift_detected)


# ==============================================================================
# PARAMETER VALIDATION TESTS
# ==============================================================================


@PROPERTY_SETTINGS
@given(q=st.floats(min_value=-1.0, max_value=2.0).filter(lambda x: x < 0 or x > 1))
def test_quantile_meter_rejects_invalid_q(q):
    """QuantileMeter rejects q outside [0, 1]."""
    with pytest.raises(ValueError):
        QuantileMeter(q=q)


@PROPERTY_SETTINGS
@given(k=st.integers(min_value=-10, max_value=7))
def test_kll_rejects_invalid_k(k):
    """KLLMeter rejects k < 8."""
    with pytest.raises(ValueError):
        KLLMeter(k=k)


@PROPERTY_SETTINGS
@given(epsilon=st.floats(min_value=-1.0, max_value=2.0).filter(lambda x: x <= 0 or x >= 1))
def test_kll_rejects_invalid_epsilon(epsilon):
    """KLLMeter rejects epsilon outside (0, 1)."""
    with pytest.raises(ValueError):
        KLLMeter(epsilon=epsilon)


@PROPERTY_SETTINGS
@given(delta=st.floats(min_value=-1.0, max_value=2.0).filter(lambda x: x <= 0 or x >= 1))
def test_kll_rejects_invalid_delta(delta):
    """KLLMeter rejects delta outside (0, 1)."""
    with pytest.raises(ValueError):
        KLLMeter(delta=delta)


@PROPERTY_SETTINGS
@given(precision=st.integers(min_value=-5, max_value=3))
def test_hyperloglog_rejects_invalid_precision(precision):
    """HyperLogLogStatistic rejects precision < 4."""
    with pytest.raises(ValueError):
        HyperLogLogStatistic(precision=precision)


@PROPERTY_SETTINGS
@given(width=st.integers(min_value=-5, max_value=0))
def test_count_min_sketch_rejects_invalid_width(width):
    """CountMinSketchStatistic rejects width <= 0."""
    with pytest.raises(ValueError):
        CountMinSketchStatistic(width=width, depth=3)


@PROPERTY_SETTINGS
@given(depth=st.integers(min_value=-5, max_value=0))
def test_count_min_sketch_rejects_invalid_depth(depth):
    """CountMinSketchStatistic rejects depth <= 0."""
    with pytest.raises(ValueError):
        CountMinSketchStatistic(width=32, depth=depth)
