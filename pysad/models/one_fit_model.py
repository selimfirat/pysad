from pysad.models.pyod_model import PYODModel


class OneFitModel(PYODModel):
    """

    Reference: https://www.andrew.cmu.edu/user/lakoglu/pubs/18-kdd-xstream.pdf
    """

    def __init__(self, model_cls, initial_X, initial_y=None, **kwargs):
        super().__init__(model_cls, **kwargs)

        self.initial_y = initial_y
        self.initial_X = initial_X
        self._fit_model()

    def fit_partial(self, X, y=None):

        return self

    def _fit_model(self):

        self.reset_model()

        if self.initial_y is None:
            self.model.fit(self.initial_y)
        else:
            self.model.fit(self.initial_X, self.initial_X)

    def score_partial(self, X):

        score = self.model.decision_function([X])[0]

        return score
