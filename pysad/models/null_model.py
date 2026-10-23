from __future__ import annotations

from typing import Any

from pysad.core.base_model import BaseModel


class NullModel(BaseModel):
    """Returns 0.5 for every instance. Use it to test a pipeline."""

    def __init__(self) -> None:
        self.labels: list[Any] = []

    def fit_partial(self, X: Any, y: Any = None) -> NullModel:
        """This method is ignored. Added for convenience.

        Args:
            X: any
            y: any

        Returns:
            object: The model itself.
        """
        return self

    def score_partial(self, X: Any) -> float:
        """Directly returns 0.5.

        Args:
            X: any (Ignored)

        Returns:
            float: Equal to `0.5`.
        """
        return 0.5
