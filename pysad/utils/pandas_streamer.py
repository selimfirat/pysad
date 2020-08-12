from pysad.utils.array_streamer import ArrayStreamer
from pysad.core.base_streamer import BaseStreamer


class PandasStreamer(BaseStreamer):
    """Simulator class to iterate dataframe(s).

    Args:
        shuffle (bool): Whether shuffle the data initially (Default=False).
    """

    def __init__(self, shuffle=False):
        super().__init__(shuffle=shuffle)

        self.array_iterator = ArrayStreamer(shuffle=shuffle)

    def iter(self, X, y=None):
        """Iterates pandas dataframes of of features and possibly labels.

        Args:
            X: Pandas Dataframe for features.
            y: Pandas dataframe for labels.
        """
        if y is None:
            for x in self.array_iterator.iter(X.to_numpy()):
                yield x
        else:
            assert len(X) == len(y)

            for x, yr in self.array_iterator.iter(X.to_numpy(), y.to_numpy()):
                yield x, yr
