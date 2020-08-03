from pysad.core.base_model import BaseModel
import numpy as np


class RandomModel(BaseModel):

    def __init__(self):


        self.labels = []

    def fit_partial(self, x, y=None):

        return self

    def score_partial(self, x):

        return np.random.uniform()
