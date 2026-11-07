import math

import numpy as np
import pytest

from pysad.statistics import FHDDMMeter


def _reference_drifts(values, window_size=100, delta=1e-6):
    # FHDDM (Corollary I of the paper) on explicit lists, which recomputes the accuracy
    # of every full window since the last drift at each value.
    epsilon = math.sqrt(math.log(1 / delta) / (2 * window_size))
    run, drifts = [], []
    for value in values:
        run.append(1 - value)
        if len(run) < window_size:
            drifts.append(False)
            continue
        means = [np.mean(run[i - window_size : i]) for i in range(window_size, len(run) + 1)]
        drift = max(means) - means[-1] >= epsilon
        drifts.append(drift)
        if drift:
            run = []

    return drifts


def _bits(probabilities, length, seed):
    rng = np.random.RandomState(seed)
    return np.concatenate([rng.rand(length) < p for p in probabilities]).astype(float)


def _run_meter(meter, values):
    drifts = []
    for value in values:
        meter.update(value)
        drifts.append(meter.drift_detected)

    return drifts


def test_example_of_the_paper_is_detected_at_the_18th_value():
    # Fig. 4 of the paper, with n = 10 and delta = 0.2. The paper lists correct
    # predictions as 1, and the meter reads errors as 1.
    correct = [1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]
    meter = FHDDMMeter(window_size=10, delta=0.2)

    drifts = _run_meter(meter, [1 - value for value in correct])

    assert meter.epsilon == pytest.approx(0.28, abs=0.005)
    assert drifts == [False] * 17 + [True]
    assert meter.get() == pytest.approx(0.6)


def test_drifts_match_the_paper_algorithm_on_explicit_lists():
    values = _bits([0.05, 0.3, 0.6], length=300, seed=0)

    drifts = _run_meter(FHDDMMeter(window_size=50, delta=1e-3), values)

    assert drifts == _reference_drifts(values, window_size=50, delta=1e-3)
    assert sum(drifts) >= 2


def test_stationary_stream_has_no_drift():
    values = _bits([0.3], length=5000, seed=0)
    meter = FHDDMMeter()

    assert not any(_run_meter(meter, values))
    assert meter.num_items == 5000
    assert meter.get() == pytest.approx(values[-100:].mean())


def test_rise_of_the_error_rate_is_detected():
    values = _bits([0.2, 0.5], length=1000, seed=0)

    drifts = _run_meter(FHDDMMeter(), values)

    first = drifts.index(True)
    assert 1000 <= first < 1100


def test_fall_of_the_error_rate_is_not_a_drift():
    values = _bits([0.5, 0.1], length=1000, seed=0)

    assert not any(_run_meter(FHDDMMeter(), values))


def test_update_after_a_drift_forgets_earlier_values():
    meter = FHDDMMeter(window_size=20)
    for value in [0.0] * 100 + [1.0] * 100:
        meter.update(value)
        if meter.drift_detected:
            break
    assert meter.drift_detected

    meter.update(1.0)

    assert not meter.drift_detected
    assert meter.num_items == 1
    assert meter.get() == 1.0


@pytest.mark.parametrize("delta", [0.0, 1.0])
def test_delta_out_of_range_is_rejected(delta):
    with pytest.raises(ValueError, match="delta"):
        FHDDMMeter(delta=delta)


def test_remove_is_not_supported():
    meter = FHDDMMeter()
    meter.update(1.0)

    with pytest.raises(NotImplementedError):
        meter.remove(1.0)
