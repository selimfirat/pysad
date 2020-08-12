from pysad.core.base_model import BaseModel


class PYODModel(BaseModel):

    def __init__(self, model_cls, **kwargs):
        """Abstract base class for PYOD models.

        Args:
            model_cls (class): The model class to be instantiated.
            **kwargs (Keyword arguments): Keyword arguments that is passed to the `model_cls`.
        """
        self.model_cls = model_cls
        self.kwargs = kwargs
        self.model = None

    def reset_model(self):
        """Removes the old model from the memory and instantiates a new one.
        """
        self.model = self.model_cls(**self.kwargs)
