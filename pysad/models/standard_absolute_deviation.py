from pysad.core.base_model import BaseModel
from pysad.statistics.average_meter import AverageMeter
from pysad.statistics.median_meter import MedianMeter
from pysad.statistics.variance_meter import VarianceMeter


class StandardAbsoluteDeviation(BaseModel):
    """The model that assigns the deviation from the mean (or median) and divides with the standard deviation. This model is based on the 3-Sigma rule described in :cite:`hochenbaum2017automatic`.

        substracted_statistic: str (Default="mean")
            The statistic to be substracted for scoring. It is either "mean" or "median".
        absolute: bool (Default=True)
            Whether to output score's absolute value.
    """

    def __init__(self, substracted_statistic="mean", absolute=True):
        self.absolute = absolute
        self.variance_meter = VarianceMeter()

        if substracted_statistic == "median":
            self.sub_meter = MedianMeter()
        elif substracted_statistic == "mean":
            self.sub_meter = AverageMeter()
        else:
            raise ValueError(
                "Unknown substracted_statistic value! Please choose median or mean.")

    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X: np.float array of shape (1,)
                The instance to fit. Note that this model is univariate.
            y: int (Default=None)
                Ignored since the model is unsupervised.

        Returns:
            self: object
                Returns the self.
        """
        assert len(X) == 1  # Only for time series

        self.variance_meter.update(X)
        self.sub_meter.update(X)

        return self

    def score_partial(self, X):
        sub = self.sub_meter.get()
        dev = self.variance_meter.get()**0.5

        score = (X - sub) / (dev + 1e-10)

        return abs(score) if self.absolute else score
