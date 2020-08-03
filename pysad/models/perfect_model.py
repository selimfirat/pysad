from pysad.core.base_model import BaseModel


class PerfectModel(BaseModel):

    def __init__(self):


        self.labels = []

    def fit_partial(self, x, y=None):
        if y is None:
            raise ValueError("y should be the true score")

        self.labels.append(y)

        return self

    def score_partial(self, x):

        score = self.labels[0]
        self.labels = self.labels[1:]

        return score
