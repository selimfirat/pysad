from pysad.models.pyod_model import PYODModel
import numpy as np


class ReferenceWindowModel(PYODModel):
    """

    Reference: https://www.andrew.cmu.edu/user/lakoglu/pubs/18-kdd-xstream.pdf
    """

    def __init__(self, model_cls, window_size, sliding_size, initial_window_X=None, initial_window_y=None):
        super().__init__(model_cls)

        self.window_size = window_size
        self.sliding_size = sliding_size

        self.cur_window_X = []
        self.cur_window_y = []

        self.reference_window_X = initial_window_X
        self.reference_window_y = initial_window_y

        if self.reference_window_X is not None:
            self._fit_model()

    def fit_partial(self, X, y=None):

        self.cur_window_X.append(X)

        if y is not None:
            self.cur_window_y.append(y)

        if self.reference_window_X is None:
            self.reference_window_X = self.cur_window_X
            self.reference_window_y = self.cur_window_y if y is not None else None
            self._fit_model()
        elif len(self.cur_window_X) % self.sliding_size == 0:
            self.reference_window_X = np.concatenate([self.reference_window_X, self.cur_window_X], axis=0)
            self.reference_window_X = self.reference_window_X[max(0, len(self.reference_window_X) - self.window_size):]

            if y is not None:
                self.reference_window_y = self.reference_window_y[max(0, len(self.reference_window_y) - self.window_size):]
                self.reference_window_y = np.concatenate([self.reference_window_y, self.cur_window_y], axis=0)

            self.cur_window = []
            self._fit_model()

        return self

    def _fit_model(self):

        self.reset_model()

        if self.reference_window_y is None:
            self.model.fit(self.reference_window_X)
        else:
            self.model.fit(self.reference_window_X, self.reference_window_y)

    def score_partial(self, X):

        score = self.model.decision_function([X])[0]

        return score
