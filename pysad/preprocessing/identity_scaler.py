import torch

from pysad.preprocessing.base_scaler import BaseScaler


class IdentityScaler(BaseScaler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_partial(self, X):

        return self

    def transform_partial(self, X):

        return X
