from models.base_model import BaseModel


class StreamLocalOutlierProbability(BaseModel):

    def __init__(self, initial_X, num_neighbors=10, extent=3, **kwargs):
        super().__init__(**kwargs)
        from PyNomaly import loop

        self.model = loop.LocalOutlierProbability(data=initial_X, extent=extent, n_neighbors=num_neighbors)

    def fit_partial(self, X, y=None):

        return self

    def score_partial(self, X):

        return self.model.stream(X)
