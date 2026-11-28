"""Shared contract tests for all running statistics in ``pysad.statistics``.

For every class in ``pysad.statistics.__all__`` these tests check that:

* ``get`` returns a finite number after many updates with the same value,
* repeated calls to ``get`` without new updates return the same value,
* repeated ``get`` calls do not change the result of a later ``update``,
* ``remove(x)`` right after ``update(x)`` restores the previous value, for statistics that support removal.

Only public methods are used. Stochastic statistics are seeded so the tests are reproducible.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import pytest

import pysad.statistics
from pysad.statistics import AverageMeter

CONSTANT = 0.0
NUM_UPDATES = 50

STATISTIC_PARAMS: dict[str, dict[str, Any]] = {
    "AbsStatistic": {"statistic_cls": AverageMeter},
    "CountMinSketchStatistic": {"random_state": 7},
    "KLLMeter": {"random_state": 7},
    "KSWINMeter": {"window_size": 16, "stat_size": 6, "random_state": 7},
    "MaxMeter": {"keep_values": True},
    "MinMeter": {"keep_values": True},
    "OnlinePeriodicityMeter": {"window_size": 8},
    "RelativeErrorQuantileMeter": {"random_state": 7},
    "RunningStatistic": {"statistic_cls": AverageMeter, "window_size": 6},
}


def _statistic_classes() -> list[tuple[str, type[Any]]]:
    """Return the concrete classes exported by ``pysad.statistics``."""
    classes: list[tuple[str, type[Any]]] = []
    for name in pysad.statistics.__all__:
        obj = getattr(pysad.statistics, name)
        if inspect.isclass(obj):
            classes.append((name, obj))
    return classes


STATISTIC_CLASSES = _statistic_classes()
STATISTIC_IDS = [stat_name for stat_name, _ in STATISTIC_CLASSES]


def _make_statistic(name: str, cls: type[Any]) -> Any:
    """Create one statistic with the parameters used by the contract tests."""
    return cls(**STATISTIC_PARAMS.get(name, {}))


def _is_removable(meter: Any) -> bool:
    """Return True when the statistic supports removing observations."""
    return bool(getattr(meter, "_removable", False))


@pytest.mark.parametrize(("name", "cls"), STATISTIC_CLASSES, ids=STATISTIC_IDS)
def test_statistic_get_returns_finite_number_on_constant_stream(name: str, cls: type[Any]) -> None:
    """``get`` returns a finite number after many identical updates.

    Args:
        name: The exported name of the statistic.
        cls: The statistic class.
    """
    meter = _make_statistic(name, cls)
    for _ in range(NUM_UPDATES):
        meter.update(CONSTANT)
    value = meter.get()
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    assert math.isfinite(float(value))


@pytest.mark.parametrize(("name", "cls"), STATISTIC_CLASSES, ids=STATISTIC_IDS)
def test_statistic_get_is_stable_on_repeated_calls(name: str, cls: type[Any]) -> None:
    """Repeated calls to ``get`` without updates return the same value.

    Args:
        name: The exported name of the statistic.
        cls: The statistic class.
    """
    meter = _make_statistic(name, cls)
    for _ in range(NUM_UPDATES):
        meter.update(CONSTANT)
    first_value = meter.get()
    second_value = meter.get()
    assert first_value == second_value


@pytest.mark.parametrize(("name", "cls"), STATISTIC_CLASSES, ids=STATISTIC_IDS)
def test_removable_statistic_remove_undoes_update(name: str, cls: type[Any]) -> None:
    """``remove(x)`` right after ``update(x)`` restores the previous value.

    Statistics that do not support removal are skipped.

    Args:
        name: The exported name of the statistic.
        cls: The statistic class.
    """
    meter = _make_statistic(name, cls)
    if not _is_removable(meter):
        pytest.skip(f"{name} does not support removal")
    meter.update(CONSTANT)
    previous_value = meter.get()
    meter.update(CONSTANT)
    meter.remove(CONSTANT)
    assert meter.get() == previous_value


@pytest.mark.parametrize(("name", "cls"), STATISTIC_CLASSES, ids=STATISTIC_IDS)
def test_statistic_get_does_not_affect_later_update(name: str, cls: type[Any]) -> None:
    """Repeated ``get`` calls do not change the result of a later ``update``.

    Args:
        name: The exported name of the statistic.
        cls: The statistic class.
    """
    meter = _make_statistic(name, cls)
    for _ in range(NUM_UPDATES):
        meter.update(CONSTANT)
    # Call get multiple times
    for _ in range(5):
        meter.get()
    # Update with a new value
    meter.update(1.0)
    # The result should be the same as if get was never called
    meter2 = _make_statistic(name, cls)
    for _ in range(NUM_UPDATES):
        meter2.update(CONSTANT)
    meter2.update(1.0)
    assert meter.get() == meter2.get()
