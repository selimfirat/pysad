import math

import numpy as np
import pytest

from pysad.statistics import DDMMeter


def _reference_states(values, warning_level=2.0, drift_level=3.0, min_num_items=30):
    # DDM (Section 3 of the paper) on explicit lists, which recomputes the error rate
    # and the minimum of p_i + s_i from the whole context at each value.
    context, states = [], []
    for value in values:
        context.append(value)
        if len(context) <= min_num_items:
            states.append((False, False))
            continue
        sums = []
        for i in range(min_num_items + 1, len(context) + 1):
            p = np.mean(context[:i])
            sums.append((p + math.sqrt(p * (1 - p) / i), p, math.sqrt(p * (1 - p) / i)))
        # The minimum moves only when the sum is strictly lower, so take the first one.
        _, p_min, s_min = min(sums, key=lambda item: item[0])
        level = sums[-1][0]
        drift = level > p_min + drift_level * s_min
        warning = not drift and level > p_min + warning_level * s_min
        states.append((drift, warning))
        if drift:
            context = []

    return states


def _bits(probabilities, length, seed):
    rng = np.random.RandomState(seed)
    return np.concatenate([rng.rand(length) < p for p in probabilities]).astype(float)


def _run_meter(meter, values):
    states = []
    for value in values:
        meter.update(value)
        states.append((meter.drift_detected, meter.warning_detected))

    return states


def test_states_match_the_paper_algorithm_on_explicit_lists():
    values = _bits([0.05, 0.3, 0.8], length=150, seed=0)

    states = _run_meter(DDMMeter(), values)

    assert states == _reference_states(values)
    assert sum(drift for drift, _ in states) >= 2


def test_stationary_stream_has_no_drift():
    values = _bits([0.3], length=5000, seed=1)
    meter = DDMMeter()

    assert not any(drift for drift, _ in _run_meter(meter, values))
    assert meter.num_items == 5000
    assert meter.get() == pytest.approx(values.mean())


def test_rise_of_the_error_rate_is_detected_after_a_warning():
    values = _bits([0.2, 0.5], length=1000, seed=0)

    states = _run_meter(DDMMeter(), values)

    drifts = [i for i, (drift, _) in enumerate(states) if drift]
    assert drifts
    assert 1000 <= drifts[0] < 1500
    assert states[drifts[0] - 1][1]
    assert not any(drift and warning for drift, warning in states)


def test_first_error_after_more_than_min_num_items_correct_values_is_a_drift():
    states = _run_meter(DDMMeter(min_num_items=50), [0.0] * 51 + [1.0])

    assert not any(drift or warning for drift, warning in states[:-1])
    assert states[-1] == (True, False)


def test_update_after_a_drift_forgets_earlier_values():
    meter = DDMMeter()
    for value in [0.0] * 100 + [1.0] * 100:
        meter.update(value)
        if meter.drift_detected:
            break
    assert meter.drift_detected

    meter.update(1.0)

    assert not meter.drift_detected
    assert meter.num_items == 1
    assert meter.get() == 1.0


def test_levels_out_of_order_are_rejected():
    with pytest.raises(ValueError, match="warning_level < drift_level"):
        DDMMeter(warning_level=3.0, drift_level=2.0)


def test_remove_is_not_supported():
    from pysad.statistics import RunningStatistic

    meter = DDMMeter()
    assert not hasattr(meter, "remove")
    with pytest.raises(TypeError, match="does not support remove"):
        RunningStatistic(DDMMeter, window_size=10)
