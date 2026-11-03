import math
import pickle

import numpy as np
import pytest

from pysad.statistics import ADWINMeter


def _reference_widths(values, delta):
    # Plain ADWIN on an explicit list, testing every split with Eq. 3.1 of the paper.
    window = []
    widths = []
    for value in values:
        window.append(value)
        while len(window) > 1:
            items = np.asarray(window)
            n = len(items)
            log_term = math.log(2.0 * math.log(n) / delta)
            n0 = np.arange(1, n)
            sum0 = items.cumsum()[:-1]
            inv_m = 1.0 / n0 + 1.0 / (n - n0)
            eps_cut = np.sqrt(2.0 * inv_m * items.var() * log_term) + 2.0 / 3.0 * inv_m * log_term
            if not (np.abs(sum0 / n0 - (items.sum() - sum0) / (n - n0)) > eps_cut).any():
                break
            window.pop(0)
        widths.append(len(window))

    return widths


def _shifting_stream():
    rng = np.random.RandomState(1)
    return np.concatenate(
        [rng.normal(0.0, 1.0, 300), rng.normal(1.5, 1.0, 300), rng.normal(-1.0, 1.0, 300)]
    )


def _run_meter(meter, values):
    drifts = []
    for i, value in enumerate(values):
        meter.update(value)
        if meter.drift_detected:
            drifts.append(i)

    return drifts


def test_stationary_stream_keeps_the_whole_window():
    values = np.random.RandomState(0).rand(2000)
    meter = ADWINMeter()

    assert _run_meter(meter, values) == []
    assert meter.width == 2000
    assert meter.get() == pytest.approx(values.mean())


def test_mean_shift_shrinks_the_window_to_the_new_mean():
    rng = np.random.RandomState(0)
    values = np.concatenate([rng.rand(1000), rng.rand(1000) + 1.0])
    meter = ADWINMeter()

    drifts = _run_meter(meter, values)

    assert drifts
    assert 1000 <= drifts[0] < 1050
    # Every pre-change item has been dropped.
    assert meter.width <= 1000
    assert meter.get() == pytest.approx(1.5, abs=0.05)


def test_window_is_always_the_most_recent_items():
    values = _shifting_stream()
    meter = ADWINMeter()

    for i, value in enumerate(values):
        meter.update(value)
        assert meter.get() == pytest.approx(values[i + 1 - meter.width : i + 1].mean())

    # The window ends as exactly the last segment, without dropping any of its items.
    assert meter.width == 300


def test_single_item_buckets_match_plain_adwin():
    # With buckets that never merge, every split is tested, as in plain ADWIN.
    values = _shifting_stream()
    meter = ADWINMeter(delta=0.01, max_buckets=len(values))

    widths = []
    for value in values:
        meter.update(value)
        widths.append(meter.width)

    assert widths == _reference_widths(values, delta=0.01)


def test_drift_flag_is_set_only_on_updates_that_shrink_the_window():
    meter = ADWINMeter()
    flags = []
    for value in [0.0] * 200 + [1.0] * 200:
        width = meter.width
        meter.update(value)
        flags.append((meter.drift_detected, meter.width <= width))

    assert all(drift == shrank for drift, shrank in flags)
    assert any(drift for drift, _ in flags)
    assert not flags[-1][0]


def test_memory_grows_logarithmically_with_the_window():
    def pickled_size(num_items):
        meter = ADWINMeter()
        for value in np.random.RandomState(0).rand(num_items):
            meter.update(value)
        assert meter.width == num_items
        return len(pickle.dumps(meter))

    # A window 16 times longer adds only a few bucket levels.
    assert pickled_size(2**14) < 2 * pickled_size(2**10)


@pytest.mark.parametrize("delta", [0.0, 1.0, -0.1])
def test_delta_outside_unit_interval_is_rejected(delta):
    with pytest.raises(ValueError, match="delta must be in"):
        ADWINMeter(delta=delta)


def test_remove_is_not_supported():
    meter = ADWINMeter()
    meter.update(1.0)

    with pytest.raises(NotImplementedError):
        meter.remove(1.0)
