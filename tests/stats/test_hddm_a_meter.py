import math

import numpy as np
import pytest

from pysad.statistics import HDDMAMeter


def _reference_states(values, drift_confidence=0.001, warning_confidence=0.005):
    # HDDM with the A-test (Fig. 1, Corollary 9 of the paper) on explicit lists, which
    # searches every prefix for the cut point instead of keeping it incrementally.
    def rejects(run, cut, confidence):
        n = len(run)
        if cut == n:
            return False
        eps = math.sqrt((n - cut) / (2 * cut * n) * math.log(1 / confidence))
        return np.mean(run) - np.mean(run[:cut]) >= eps

    run, cut, states = [], 0, []
    for value in values:
        run.append(value)
        bounds = [
            np.mean(run[:i]) + math.sqrt(math.log(1 / drift_confidence) / (2 * i))
            for i in range(1, len(run) + 1)
        ]
        # The cut point moves forward on ties, so take the last minimum.
        cut = len(bounds) - int(np.argmin(bounds[::-1]))
        drift = rejects(run, cut, drift_confidence)
        warning = not drift and rejects(run, cut, warning_confidence)
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
    values = _bits([0.2, 0.6, 0.1], length=150, seed=0)

    states = _run_meter(HDDMAMeter(), values)

    assert states == _reference_states(values)
    assert any(drift for drift, _ in states)


def test_stationary_stream_has_no_drift():
    values = np.random.RandomState(0).rand(5000)
    meter = HDDMAMeter()

    assert not any(drift for drift, _ in _run_meter(meter, values))
    assert meter.num_items == 5000
    assert meter.get() == pytest.approx(values.mean())


def test_rise_of_the_mean_is_detected_after_a_warning():
    values = _bits([0.2, 0.6], length=1000, seed=0)

    states = _run_meter(HDDMAMeter(), values)

    drifts = [i for i, (drift, _) in enumerate(states) if drift]
    warnings = [i for i, (_, warning) in enumerate(states) if warning]
    assert drifts
    assert 1000 <= drifts[0] < 1100
    assert warnings and warnings[0] < drifts[0]
    assert not any(drift and warning for drift, warning in states)


def test_update_after_a_drift_forgets_earlier_values():
    meter = HDDMAMeter()
    for value in [0.0] * 100 + [1.0] * 100:
        meter.update(value)
        if meter.drift_detected:
            break
    assert meter.drift_detected

    meter.update(0.5)

    assert not meter.drift_detected
    assert meter.num_items == 1
    assert meter.get() == 0.5


def test_fall_of_the_mean_is_detected_only_when_two_sided():
    values = _bits([0.6, 0.2], length=1000, seed=0)

    one_sided = _run_meter(HDDMAMeter(), values)
    two_sided = _run_meter(HDDMAMeter(two_sided=True), values)

    assert not any(drift for drift, _ in one_sided)
    assert 1000 <= next(i for i, (drift, _) in enumerate(two_sided) if drift) < 1100


@pytest.mark.parametrize(
    ("drift_confidence", "warning_confidence"), [(0.0, 0.005), (0.005, 0.001), (0.001, 1.5)]
)
def test_confidences_out_of_order_are_rejected(drift_confidence, warning_confidence):
    with pytest.raises(ValueError, match="drift_confidence < warning_confidence"):
        HDDMAMeter(drift_confidence=drift_confidence, warning_confidence=warning_confidence)


def test_remove_is_not_supported():
    meter = HDDMAMeter()
    meter.update(1.0)

    with pytest.raises(NotImplementedError):
        meter.remove(1.0)
