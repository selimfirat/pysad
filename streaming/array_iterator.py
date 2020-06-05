from streaming.base_iterator import BaseIterator
import numpy as np


class ArrayIterator(BaseIterator):

    def iter(self, X, y=None):
        indices = list(range(len(X)))
        if self.shuffle:
            np.random.shuffle(indices)

        if y is None:
            for i in indices:
                yield X[i]
        else:
            assert len(X) == len(y)
            for i in indices:
                yield X[i], y[i]
