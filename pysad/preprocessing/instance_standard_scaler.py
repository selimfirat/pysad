from pysad.preprocessing.base_scaler import BaseScaler


class InstanceStandardScaler(BaseScaler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_partial(self, X):

        return self

    def transform_partial(self, X):
        X_mean = X.mean(dim=1, keepdim=True)
        X_std = X.std(dim=1, keepdim=True)

        return (X - X_mean.expand_as(X)) / X_std.expand_as(X)
