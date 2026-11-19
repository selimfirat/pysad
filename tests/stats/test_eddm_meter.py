import numpy as np
import pytest

from pysad.statistics import EDDMMeter


def _reference_states(values, warning_threshold=0.95, drift_threshold=0.9, min_num_errors=30):
    # EDDM (Section 3.2 of the paper) on explicit lists of the distances between errors.
    distances, position, last_error, max_level, warning, states = [], 0, 0, 0.0, False, []
    for value in values:
        position += 1
        drift = False
        if value:
            distances.append(position - last_error)
            last_error = position
            if len(distances) >= min_num_errors:
                level = np.mean(distances) + 2 * np.std(distances)
                max_level = max(max_level, level)
                drift = level / max_level < drift_threshold
                warning = not drift and level / max_level < warning_threshold
        states.append((drift, warning))
        if drift:
            distances, position, last_error, max_level, warning = [], 0, 0, 0.0, False

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


@pytest.mark.parametrize(
    ("warning_threshold", "drift_threshold", "min_num_errors"), [(0.95, 0.9, 30), (0.9, 0.8, 50)]
)
def test_states_match_the_paper_algorithm_on_explicit_lists(
    warning_threshold, drift_threshold, min_num_errors
):
    values = _bits([0.05, 0.3, 0.1], length=1000, seed=0)
    meter = EDDMMeter(
        warning_threshold=warning_threshold,
        drift_threshold=drift_threshold,
        min_num_errors=min_num_errors,
    )

    states = _run_meter(meter, values)

    assert states == _reference_states(values, warning_threshold, drift_threshold, min_num_errors)
    assert any(drift for drift, _ in states)
    assert any(warning for _, warning in states)


def test_rise_of_the_error_rate_is_detected_after_a_warning():
    values = _bits([0.05, 0.3], length=2000, seed=0)

    states = _run_meter(EDDMMeter(), values)

    drift = next(i for i, (drift, _) in enumerate(states) if drift and i >= 2000)
    warnings = [i for i, (_, warning) in enumerate(states[2000:drift]) if warning]
    assert drift < 2100
    assert warnings
    assert not any(drift and warning for drift, warning in states)


def test_no_flag_until_an_error_after_min_num_errors():
    values = _bits([0.05, 0.3], length=2000, seed=0)
    meter = EDDMMeter(min_num_errors=50)

    flags_at = []
    for value in values:
        meter.update(value)
        if meter.drift_detected or meter.warning_detected:
            flags_at.append(meter.num_errors)

    assert flags_at
    assert min(flags_at) > 50


def test_warning_stays_on_between_errors():
    values = _bits([0.05, 0.3], length=2000, seed=0)
    meter = EDDMMeter()
    for value in values:
        meter.update(value)
        if meter.warning_detected:
            break
    assert meter.warning_detected

    meter.update(0.0)

    assert meter.warning_detected


def test_update_after_a_drift_forgets_earlier_values():
    meter = EDDMMeter()
    for value in ([0.0] * 9 + [1.0]) * 30 + [1.0] * 100:
        meter.update(value)
        if meter.drift_detected:
            break
    assert meter.drift_detected

    meter.update(0.0)
    meter.update(1.0)

    assert not meter.drift_detected
    assert meter.num_items == 2
    assert meter.num_errors == 1
    assert meter.get() == 2.0


def test_get_is_the_mean_distance_between_errors():
    meter = EDDMMeter()
    assert meter.get() == 0.0

    for value in [0, 1, 0, 0, 1, 1]:
        meter.update(value)

    assert meter.get() == pytest.approx((2 + 3 + 1) / 3)


@pytest.mark.parametrize(
    ("warning_threshold", "drift_threshold"), [(0.95, 0.0), (0.9, 0.95), (1.5, 0.9)]
)
def test_thresholds_out_of_order_are_rejected(warning_threshold, drift_threshold):
    with pytest.raises(ValueError, match="drift_threshold <= warning_threshold"):
        EDDMMeter(warning_threshold=warning_threshold, drift_threshold=drift_threshold)


def test_remove_is_not_supported():
    from pysad.statistics import RunningStatistic

    meter = EDDMMeter()
    assert not hasattr(meter, "remove")
    with pytest.raises(TypeError, match="does not support remove"):
        RunningStatistic(EDDMMeter, window_size=10)
