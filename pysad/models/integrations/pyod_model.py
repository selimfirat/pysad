from __future__ import annotations

from typing import Any

from pysad.core.base_model import BaseModel


class PYODModel(BaseModel):
    """Abstract base class for the wrappers that run PyOD batch models on a stream.

    ``ReferenceWindowModel`` and ``OneFitModel`` extend this class. It stores ``model_cls`` and its keyword arguments, and ``reset_model`` builds a new ``model_cls(**kwargs)`` in place of the old model.

    Args:
        model_cls (class): The model class to be instantiated.
        **kwargs (Keyword arguments): Keyword arguments that are passed to the ``model_cls``.
    """

    def __init__(self, model_cls: type[Any], **kwargs: Any) -> None:
        self.model_cls = model_cls
        self.kwargs = kwargs
        self.model: Any = None

    def reset_model(self) -> None:
        """Removes the old model from the memory and instantiates a new one."""
        self.model = self.model_cls(**self.kwargs)
