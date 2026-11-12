from __future__ import annotations

import inspect
from collections import deque
from typing import Any, cast

from pysad.core.base_statistic import BaseStatistic, RemovableStatistic


class RunningStatistic(BaseStatistic):
    """Wraps a removable streaming statistic to track values over a fixed sliding window.

    Args:
        statistic_cls (class): Removable statistic class to instantiate and window.
        window_size (int): The window size. Must be at least 1.
        **kwargs (Keyword arguments): Keyword arguments passed to ``statistic_cls``.

    Raises:
        ValueError: If ``window_size`` is less than 1.
        TypeError: If the instantiated statistic does not support ``remove``.
    """

    def __init__(self, statistic_cls: type[BaseStatistic], window_size: int, **kwargs: Any) -> None:
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")

        init_params = inspect.signature(statistic_cls.__init__).parameters
        if "keep_values" in init_params and "keep_values" not in kwargs:
            kwargs["keep_values"] = True
        elif (
            "keep_values" not in kwargs
            and "statistic_cls" in kwargs
            and inspect.isclass(kwargs["statistic_cls"])
            and "keep_values" in inspect.signature(kwargs["statistic_cls"].__init__).parameters
        ):
            kwargs["keep_values"] = True

        self.statistic_cls = statistic_cls
        statistic = self.statistic_cls(**kwargs)
        removable = getattr(statistic, "_removable", callable(getattr(statistic, "remove", None)))
        if not removable:
            raise TypeError(
                f"{statistic_cls.__name__} does not support remove() and cannot be wrapped in RunningStatistic."
            )
        self.statistic = cast(RemovableStatistic, statistic)

        self.window_size = window_size
        self.window: deque[float] = deque()

    def update(self, num: float) -> RunningStatistic:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            RunningStatistic: self.
        """
        self.window.append(num)
        self.statistic.update(num)

        if len(self.window) > self.window_size:
            self.statistic.remove(self.window.popleft())

        return self

    def get(self) -> float:
        """Obtains the tracked windowed statistic.

        Returns:
            float: The statistic.
        """
        return self.statistic.get()
