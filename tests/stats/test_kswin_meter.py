import math

import numpy as np
import pytest
from scipy import stats

from pysad.statistics import KSWINMeter


def _spread_change(seed=0):
    # The mean stays 0, the standard deviation changes from 1 to 3 at index 1000.
    rng = np.random.RandomState(seed)
    return np.concatenate([rng.normal(0, 1, 1000), rng.normal(0, 3, 1000)])


def _run_meter(meter, values):
    drifts, dists = [], []
    for i, value in enumerate(values):
        meter.update(value)
        dists.append(meter.get())
        if meter.drift_detected:
            drifts.append(i)

    return drifts, dists


def _reference(values, alpha, window_size, stat_size, seed):
    # KSWIN on an explicit list, with SciPy's KS statistic and the threshold of Eq. 5.
    rng = np.random.RandomState(seed)
    threshold = math.sqrt(-math.log(alpha) / stat_size)
    window, drifts, dists, dist = [], [], [], 0.0
    for i, value in enumerate(values):
        window = (window + [value])[-window_size:]
        if len(window) == window_size:
            items = np.asarray(window)
            older = items[rng.choice(window_size - stat_size, size=stat_size, replace=False)]
            dist = stats.ks_2samp(older, items[-stat_size:]).statistic
            if dist > threshold:
                drifts.append(i)
                window = window[-stat_size:]
        dists.append(dist)

    return drifts, dists


def test_matches_reference_kswin():
    values = _spread_change()
    meter = KSWINMeter(alpha=0.001, window_size=100, stat_size=30, random_state=3)

    drifts, dists = _run_meter(meter, values)
    ref_drifts, ref_dists = _reference(values, alpha=0.001, window_size=100, stat_size=30, seed=3)

    assert drifts == ref_drifts
    np.testing.assert_allclose(dists, ref_dists)


def test_stationary_stream_has_no_drift():
    meter = KSWINMeter(random_state=0)

    drifts, dists = _run_meter(meter, np.random.RandomState(1).normal(size=5000))

    assert drifts == []
    assert dists[298] == 0.0
    assert 0.0 < dists[299] <= 1.0


@pytest.mark.parametrize("seed", range(5))
def test_mean_shift_is_detected_soon_after_it_starts(seed):
    rng = np.random.RandomState(seed)
    values = np.concatenate([rng.normal(0, 1, 1000), rng.normal(3, 1, 1000)])
    meter = KSWINMeter(random_state=seed)

    drifts, _ = _run_meter(meter, values)

    assert drifts
    assert 1000 <= drifts[0] < 1050


def test_window_keeps_the_newest_values_after_a_drift():
    meter = KSWINMeter(window_size=100, stat_size=30, random_state=0)

    drifts, dists = _run_meter(meter, [0.0] * 100 + [1.0] * 400)

    # The window is full again only after window_size - stat_size more values.
    assert drifts[0] < 130
    assert dists[drifts[0] + 1 : drifts[0] + 70] == [dists[drifts[0]]] * 69
    assert (np.diff(drifts) >= 70).all()


def test_same_seed_gives_same_distances():
    values = _spread_change()

    _, first = _run_meter(KSWINMeter(random_state=5), values)
    _, second = _run_meter(KSWINMeter(random_state=5), values)

    assert first == second


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_alpha_outside_unit_interval_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha must be in"):
        KSWINMeter(alpha=alpha)


def test_window_smaller_than_two_samples_is_rejected():
    with pytest.raises(ValueError, match="window_size must be at least 2 \\* stat_size"):
        KSWINMeter(window_size=59, stat_size=30)


def test_window_of_exactly_two_samples_tests_when_full():
    meter = KSWINMeter(window_size=60, stat_size=30, random_state=0)
    for i in range(60):
        meter.update(float(i % 7))

    assert 0.0 <= meter.get() <= 1.0


def test_remove_is_not_supported():
    meter = KSWINMeter()
    meter.update(1.0)

    with pytest.raises(NotImplementedError):
        meter.remove(1.0)
