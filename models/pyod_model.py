from abc import ABC

from models.base_model import BaseModel


class PYODModel(ABC, BaseModel):

    def __init__(self, model_cls, **kwargs):
        super().__init__(**kwargs)

        self.model_cls = model_cls
        self.model = self.reset_model()

    def reset_model(self):

        return self.model_cls(self.kwargs)
