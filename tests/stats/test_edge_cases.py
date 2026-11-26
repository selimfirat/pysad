"""Edge-case robustness tests for streaming statistics and drift detectors.

These tests complement tests/stats/test_contracts.py, which pins the documented
happy-path behavior (a finite ``get`` on a constant stream, stable repeated
``get`` calls, and ``remove`` undoing ``update``). These tests cover the
boundary inputs: ``get`` with no updates, ``get`` after a single update, and
drift meters that stay silent on a constant stream and clear their alarm flag
on the next update.

Only public methods and attributes are used. Every statistic is exercised with
its default constructor unless a non-default argument is required.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

import pysad.statistics
from pysad.statistics import AverageMeter

CONSTANT = 0.0
NUM_UPDATES = 25


def _statistic_classes() -> list[tuple[str, type[Any]]]:
    """Return the concrete classes exported by ``pysad.statistics``."""
    classes: list[tuple[str, type[Any]]] = []
    for name in pysad.statistics.__all__:
        obj = getattr(pysad.statistics, name)
        if isinstance(obj, type):
            classes.append((name, obj))
    return classes


STATISTIC_CLASSES = _statistic_classes()
STATISTIC_IDS = [name for name, _ in STATISTIC_CLASSES]

# Constructor arguments for statistics whose defaults cannot be instantiated
# directly. Every other statistic uses its default constructor.
STATISTIC_PARAMS: dict[str, dict[str, Any]] = {
    "AbsStatistic": {"statistic_cls": AverageMeter},
    "RunningStatistic": {"statistic_cls": AverageMeter, "window_size": 4},
}


def _make_statistic(name: str, cls: type[Any]) -> Any:
    """Create one statistic with the parameters used by the edge-case tests."""
    return cls(**STATISTIC_PARAMS.get(name, {}))


def _is_drift_meter(meter: Any) -> bool:
    """Return True when the statistic exposes a public ``drift_detected`` flag."""
    return hasattr(meter, "drift_detected")


DRIFT_METER_CLASSES: list[tuple[str, type[Any]]] = []
DRIFT_METER_IDS: list[str] = []
for _name, _cls in STATISTIC_CLASSES:
    _meter = _make_statistic(_name, _cls)
    if _is_drift_meter(_meter):
        DRIFT_METER_CLASSES.append((_name, _cls))
        DRIFT_METER_IDS.append(_name)


@pytest.mark.parametrize(("name", "cls"), STATISTIC_CLASSES, ids=STATISTIC_IDS)
def test_get_before_any_update_returns_finite(name: str, cls: type[Any]) -> None:
    """``get`` with zero updates returns a finite number, not -inf, inf, or NaN.

    Args:
        name: The exported name of the statistic.
        cls: The statistic class.
    """
    meter = _make_statistic(name, cls)
    value = meter.get()
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    assert math.isfinite(float(value))


@pytest.mark.parametrize(("name", "cls"), STATISTIC_CLASSES, ids=STATISTIC_IDS)
def test_get_after_single_update_returns_finite(name: str, cls: type[Any]) -> None:
    """``get`` after one update returns a finite number.

    Args:
        name: The exported name of the statistic.
        cls: The statistic class.
    """
    meter = _make_statistic(name, cls)
    meter.update(CONSTANT)
    value = meter.get()
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    assert math.isfinite(float(value))


@pytest.mark.parametrize(("name", "cls"), STATISTIC_CLASSES, ids=STATISTIC_IDS)
def test_repeated_get_with_no_updates_is_stable(name: str, cls: type[Any]) -> None:
    """Repeated ``get`` calls with no updates return the same value.

    Args:
        name: The exported name of the statistic.
        cls: The statistic class.
    """
    meter = _make_statistic(name, cls)
    first_value = meter.get()
    second_value = meter.get()
    assert first_value == second_value


@pytest.mark.parametrize(("name", "cls"), DRIFT_METER_CLASSES, ids=DRIFT_METER_IDS)
def test_constant_stream_never_reports_drift(name: str, cls: type[Any]) -> None:
    """A constant stream never sets the drift flag on any update.

    Args:
        name: The exported name of the drift meter.
        cls: The drift meter class.
    """
    meter = _make_statistic(name, cls)
    for _ in range(NUM_UPDATES):
        meter.update(CONSTANT)
        assert meter.drift_detected is False


@pytest.mark.parametrize(("name", "cls"), DRIFT_METER_CLASSES, ids=DRIFT_METER_IDS)
def test_drift_flag_clears_on_next_update(name: str, cls: type[Any]) -> None:
    """Setting the drift flag and updating again clears it on a constant stream.

    Args:
        name: The exported name of the drift meter.
        cls: The drift meter class.
    """
    meter = _make_statistic(name, cls)
    meter.drift_detected = True
    meter.update(CONSTANT)
    assert meter.drift_detected is False
