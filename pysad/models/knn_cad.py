from pysad.core.base_model import BaseModel
import numpy as np


class KNNCAD(BaseModel):
    """
    https://github.com/numenta/NAB/tree/master/nab/detectors/knncad
    https://arxiv.org/abs/1608.04585
    Implementation at https://github.com/numenta/NAB/blob/master/nab/detectors/knncad/knncad_detector.py
    """
    def __init__(self, probationary_period, **kwargs):
        super().__init__(**kwargs)

        self.buf = []
        self.training = []
        self.calibration = []
        self.scores = []
        self.record_count = 0
        self.pred = -1
        self.k = 27
        self.dim = 19
        self.sigma = np.diag(np.ones(self.dim))

        self.probationaryPeriod = probationary_period

    def _metric(self,a,b):
        diff = a-np.array(b)

        return np.dot(np.dot(diff,self.sigma),diff.T)

    def _ncm(self,item, item_in_array=False):
        arr = [self._metric(x,item) for x in self.training]

        return np.sum(np.partition(arr, self.k+item_in_array)[:self.k+item_in_array])

    def fit_partial(self, x, y=None):
        self.buf.append(x)
        self.record_count += 1
        if len(self.buf) < self.dim:
            return self

        new_item = self.buf[-self.dim:]
        if self.record_count < self.probationaryPeriod:
            self.training.append(new_item)
        else:
            ost = self.record_count % self.probationaryPeriod
            if ost == 0 or ost == int(self.probationaryPeriod / 2):
                try:
                    self.sigma = np.linalg.inv(np.dot(np.array(self.training).T, self.training))
                except np.linalg.linalg.LinAlgError:
                    print('Singular Matrix at record', self.record_count)
            if len(self.scores) == 0:
                self.scores = [self._ncm(v, True) for v in self.training]

            new_score = self._ncm(new_item)

            if self.record_count >= 2 * self.probationaryPeriod:
                self.training.pop(0)
                self.training.append(self.calibration.pop(0))

            self.scores.pop(0)
            self.calibration.append(new_item)
            self.scores.append(new_score)

        return self

    def score_partial(self, x):
        if len(self.buf) < self.dim or self.record_count < self.probationaryPeriod:
            return 0.0

        new_item = self.buf[-self.dim:]

        new_score = self._ncm(new_item)
        result = 1. * len(np.where(np.array(self.scores) < new_score)[0]) / len(self.scores)

        if self.pred > 0:
            self.pred -= 1
            return 0.5
        elif result >= 0.9965:
            self.pred = int(self.probationaryPeriod / 5)

        return result
