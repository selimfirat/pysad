import scipy
import numpy as np
from pysad.core.base_model import BaseModel
from pysad.utils.window import Window


class ExactStorm(BaseModel):

    def __init__(self, W=10000, R=0.1, **kwargs):
        super().__init__(**kwargs)

        self.R = R
        self.window = Window(window_size=W)

    def fit_partial(self, X, y=None):
        self.window.update(X)

        return self

    def score_partial(self, X):

        dists = scipy.spatial.distance.cdist(self.window.get(), [X])

        return 1 - np.mean(dists < self.R)
