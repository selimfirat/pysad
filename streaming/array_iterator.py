from streaming.base_iterator import BaseIterator


class ArrayIterator(BaseIterator):

    def iter(self, X, y=None):

        if y is None:
            for i in range(len(X)):
                yield X[i]
        else:
            assert len(X) == len(y)
            for i in range(len(X)):
                yield X[i], y[i]
