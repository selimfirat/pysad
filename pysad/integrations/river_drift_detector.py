from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from river.base import DriftDetector
else:
    try:
        from river.base import DriftDetector
    except ImportError:  # pragma: no cover
        DriftDetector = object

from pysad.core.base_statistic import BaseStatistic


class RiverDriftDetector(DriftDetector):
    """Exposes a PySAD drift meter through River's drift detector interface.

    Requires River 0.26.1 or newer.

    ``update`` calls the wrapped meter's ``update`` method; ``drift_detected``
    reflects the meter's ``drift_detected`` attribute. The meter must have a
    ``drift_detected`` attribute (e.g., :class:`~pysad.statistics.PageHinkleyMeter`
    or :class:`~pysad.statistics.CUSUMMeter`).

    ``meter_factory`` must return a new, unfitted PySAD drift meter on each call.
    Use a meter class for default parameters, or ``functools.partial`` for custom
    ones. The created meter is available as ``meter_``. River's default ``clone()``
    creates an unfitted meter; ``clone(include_attributes=True)`` copies the
    fitted meter.

    Args:
        meter_factory (callable): A zero-argument callable that creates a fresh PySAD drift meter.
    """

    def __init__(self, meter_factory: Callable[[], BaseStatistic]) -> None:
        super().__init__()
        self.meter_factory = meter_factory
        self.meter_ = meter_factory()

    def update(self, x: int | float) -> None:
        """Updates the drift detector with a single data point.

        Args:
            x (int or float): The input value.
        """
        self.meter_.update(x)
        self._drift_detected = getattr(self.meter_, "drift_detected", False)

    def _reset(self) -> None:
        """Reset the detector's state."""
        super()._reset()
        self.meter_ = self.meter_factory()
