import torch

from seqad.preprocessing.base_scaler import BaseScaler


class InstanceUnitNormScaler(BaseScaler):

    def __init__(self, pow=2, **kwargs):
        super().__init__(**kwargs)
        self.pow = pow

    def fit_partial(self, X):

        return self

    def transform_partial(self, X):

        X_norm = X.norm(p=self.pow, dim=1, keepdim=True)

        return X / X_norm.expand_as(X)
