from pysad.core.base_model import BaseModel
from pysad.statistics.average_meter import AverageMeter
from pysad.statistics.median_meter import MedianMeter
from pysad.statistics.running_statistic import RunningStatistic
from pysad.statistics.variance_meter import VarianceMeter
from pysad.utils.validation import check_univariate


class StandardAbsoluteDeviation(BaseModel):
    """Scores each value by its deviation from a mean or median, divided by the running standard deviation.

    This model is based on the 3-Sigma rule described in :cite:`hochenbaum2017automatic`.

    This is a streaming standard-deviation score component, not the paper's
    S-ESD method. It does not apply STL decomposition or generalized ESD
    internally. Use :class:`pysad.models.SeasonalESD` for the paper's
    modified-STL plus standard ESD detector.

    Args:
        subtracted_statistic (str): The statistic to be subtracted for scoring. It is either "mean" or "median". (Default="mean").
        absolute (bool): Whether to output score's absolute value. (Default=True).
        window_size (int): Number of recent values used for the median when ``subtracted_statistic="median"``. Must be at least 1 (Default=500). The variance remains a running statistic over all fitted values. This parameter has no effect when ``subtracted_statistic="mean"``.

    Raises:
        ValueError: If ``window_size`` is less than 1 or ``subtracted_statistic`` is not "mean" or "median".
    """

    def __init__(self, subtracted_statistic="mean", absolute=True, window_size=500):
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")

        self.absolute = absolute
        self.variance_meter = VarianceMeter()
        self.window_size = window_size

        if subtracted_statistic == "median":
            self.sub_meter = RunningStatistic(MedianMeter, window_size=window_size)
        elif subtracted_statistic == "mean":
            self.sub_meter = AverageMeter()
        else:
            raise ValueError("Unknown subtracted_statistic value! Please choose median or mean.")

    def fit_partial(self, X, y=None):
        """Updates the model with the next instance.

        Args:
            X (np.float64 array of shape (1,)): The instance to fit. Note that this model is univariate.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        val = check_univariate(X, "StandardAbsoluteDeviation")
        self.variance_meter.update(val)
        self.sub_meter.update(val)

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (1,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if self.variance_meter.count == 0:  # there is no statistic before the first value
            return 0.0

        val = check_univariate(X, "StandardAbsoluteDeviation")
        sub = self.sub_meter.get()
        dev = self.variance_meter.get() ** 0.5

        score = (val - sub) / (dev + 1e-10)

        return float(abs(score) if self.absolute else score)
