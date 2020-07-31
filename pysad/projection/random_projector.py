from abc import ABC, abstractmethod
import sklearn
from pysad.projection.base_projector import BaseProjector


class BaseSKLearnProjector(ABC, BaseProjector):

    def __init__(self, n_components, **kwargs):
        super().__init__(n_components, **kwargs)


    @property
    @abstractmethod
    def projector(self):
        pass

    def fit_partial(self, X):

        return self

    def transform_partial(self, X):
        x = X.reshape(1, -1)

        return self.projector.fit_transform(x).reshape(-1)


class GaussianRandomProjector(BaseSKLearnProjector):

    def __init__(self, n_components='auto', *, eps=0.1, **kwargs):
        super().__init__(n_components, **kwargs)

        self.projector = sklearn.random_projection.GaussianRandomProjection(n_components=n_components, eps=eps)


class SparseRandomProjector(BaseSKLearnProjector):

    def __init__(self, n_components='auto', density="auto", eps=0.1, **kwargs):
        super().__init__(n_components, **kwargs)

        self.projector = sklearn.random_projection.SparseRandomProjection(n_components=n_components, density=density, eps=eps, dense_output=True)
