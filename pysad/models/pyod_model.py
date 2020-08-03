from abc import ABC
from pysad.core.base_model import BaseModel


class PYODModel(BaseModel, ABC):

    def __init__(self, model_cls):
        super().__init__()

        self.model_cls = model_cls
        self.model = None

    def reset_model(self):

        self.model = self.model_cls(**self.kwargs)

