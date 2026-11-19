import math

import numpy as np
import pytest

from pysad.statistics import HDDMWMeter


def _ewma(values, alpha):
    # The EWMA of values and the sum of its squared weights, from the explicit weights.
    n = len(values)
    weights = alpha * (1 - alpha) ** np.arange(n - 1, -1, -1.0)
    weights[0] = (1 - alpha) ** (n - 1)
    return float(weights @ values), float(weights @ weights)


def _reference_states(values, alpha=0.05, drift_confidence=0.001, warning_confidence=0.005):
    # HDDM with the W-test (Fig. 1, Corollary 5 of the paper) on explicit lists, which
    # searches every prefix for the cut point and computes each EWMA from its weights.
    def rejects(before, after, confidence):
        eps = math.sqrt((before[1] + after[1]) / 2 * math.log(1 / confidence))
        return after[0] - before[0] > eps

    run, states = [], []
    for value in values:
        run.append(value)
        prefixes = [_ewma(np.asarray(run[:i]), alpha) for i in range(1, len(run) + 1)]
        bounds = [z + math.sqrt(d / 2 * math.log(1 / drift_confidence)) for z, d in prefixes]
        # The cut point moves forward on ties, so take the last minimum.
        cut = len(bounds) - int(np.argmin(bounds[::-1]))
        drift = warning = False
        if cut < len(run):
            after = _ewma(np.asarray(run[cut:]), alpha)
            drift = rejects(prefixes[cut - 1], after, drift_confidence)
            warning = not drift and rejects(prefixes[cut - 1], after, warning_confidence)
        states.append((drift, warning))
        if drift:
            run = []

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
    values = _bits([0.2, 0.7, 0.1], length=120, seed=0)

    states = _run_meter(HDDMWMeter(alpha=0.1), values)

    assert states == _reference_states(values, alpha=0.1)
    assert any(drift for drift, _ in states)


def test_get_is_the_ewma_of_the_values_since_the_last_drift():
    values = np.random.RandomState(0).rand(300)
    meter = HDDMWMeter(alpha=0.2)

    _run_meter(meter, values)

    assert meter.get() == pytest.approx(_ewma(values, alpha=0.2)[0])


def test_stationary_stream_has_no_drift():
    values = np.random.RandomState(0).rand(5000)

    assert not any(drift for drift, _ in _run_meter(HDDMWMeter(), values))


def test_rise_of_the_mean_is_detected_after_a_warning():
    values = _bits([0.2, 0.6], length=1000, seed=0)

    states = _run_meter(HDDMWMeter(), values)

    drifts = [i for i, (drift, _) in enumerate(states) if drift]
    warnings = [i for i, (_, warning) in enumerate(states) if warning]
    assert drifts
    assert 1000 <= drifts[0] < 1100
    assert warnings and warnings[0] < drifts[0]
    assert not any(drift and warning for drift, warning in states)


def test_update_after_a_drift_forgets_earlier_values():
    meter = HDDMWMeter()
    for value in [0.0] * 100 + [1.0] * 100:
        meter.update(value)
        if meter.drift_detected:
            break
    assert meter.drift_detected

    meter.update(0.5)

    assert not meter.drift_detected
    assert meter.get() == 0.5


def test_fall_of_the_mean_is_detected_only_when_two_sided():
    values = _bits([0.6, 0.2], length=1000, seed=0)

    one_sided = _run_meter(HDDMWMeter(), values)
    two_sided = _run_meter(HDDMWMeter(two_sided=True), values)

    assert not any(drift for drift, _ in one_sided)
    assert 1000 <= next(i for i, (drift, _) in enumerate(two_sided) if drift) < 1100


@pytest.mark.parametrize(
    ("drift_confidence", "warning_confidence"), [(0.0, 0.005), (0.005, 0.001), (0.001, 1.5)]
)
def test_confidences_out_of_order_are_rejected(drift_confidence, warning_confidence):
    with pytest.raises(ValueError, match="drift_confidence < warning_confidence"):
        HDDMWMeter(drift_confidence=drift_confidence, warning_confidence=warning_confidence)


@pytest.mark.parametrize("alpha", [0.0, 1.5])
def test_alpha_outside_unit_interval_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha must be in"):
        HDDMWMeter(alpha=alpha)


def test_remove_is_not_supported():
    from pysad.statistics import RunningStatistic

    meter = HDDMWMeter()
    assert not hasattr(meter, "remove")
    with pytest.raises(TypeError, match="does not support remove"):
        RunningStatistic(HDDMWMeter, window_size=10)
