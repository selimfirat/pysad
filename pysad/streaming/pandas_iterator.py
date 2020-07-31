from streaming.array_iterator import ArrayIterator
from streaming.base_iterator import BaseIterator


class PandasIterator(BaseIterator):

    def __init__(self, shuffle=False, **kwargs):
        super().__init__(shuffle=shuffle, **kwargs)

        self.array_iterator = ArrayIterator(shuffle=shuffle, **kwargs)

    def iter(self, X, y=None):

        if y is None:
            for x in self.array_iterator.iter(X.to_numpy()):
                yield x
        else:
            assert len(X) == len(y)

            for x, yr in self.array_iterator.iter(X.to_numpy(), y.to_numpy()):
                yield x, yr
