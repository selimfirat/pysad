import warnings

from pysad.core.base_model import BaseModel
from pysad.statistics.average_meter import AverageMeter
from pysad.statistics.median_meter import MedianMeter
from pysad.statistics.variance_meter import VarianceMeter

_UNSET = object()


class StandardAbsoluteDeviation(BaseModel):
    """The model that assigns the deviation from the mean (or median) and divides with the standard deviation. This model is based on the 3-Sigma rule described in :cite:`hochenbaum2017automatic`.

        This is a streaming standard-deviation score component, not the paper's
        S-ESD method. It does not apply STL decomposition or generalized ESD
        internally. Use :class:`pysad.models.SeasonalESD` for the paper's
        modified-STL plus standard ESD detector.

        Args:
            subtracted_statistic (str): The statistic to be subtracted for scoring. It is either "mean" or "median". (Default="mean").
            absolute (bool): Whether to output score's absolute value. (Default=True).

        .. deprecated::
            The ``substracted_statistic`` keyword is deprecated.
            Use ``subtracted_statistic`` instead.
    """

    def __init__(self, subtracted_statistic=_UNSET, absolute=True,
                 *, substracted_statistic=_UNSET):
        if substracted_statistic is not _UNSET:
            warnings.warn(
                "The 'substracted_statistic' parameter is deprecated. "
                "Use 'subtracted_statistic' instead.",
                FutureWarning,
                stacklevel=2,
            )
            if subtracted_statistic is not _UNSET:
                raise TypeError(
                    "Cannot specify both 'subtracted_statistic' and "
                    "'substracted_statistic'."
                )
            subtracted_statistic = substracted_statistic

        if subtracted_statistic is _UNSET:
            subtracted_statistic = "mean"

        self.absolute = absolute
        self.variance_meter = VarianceMeter()

        if subtracted_statistic == "median":
            self.sub_meter = MedianMeter()
        elif subtracted_statistic == "mean":
            self.sub_meter = AverageMeter()
        else:
            raise ValueError(
                "Unknown subtracted_statistic value! Please choose median or mean.")

    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X (np.float64 array of shape (1,)): The instance to fit. Note that this model is univariate.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: Returns the self.
        """
        if len(X) != 1:
            raise ValueError("StandardAbsoluteDeviation supports univariate inputs.")

        self.variance_meter.update(X)
        self.sub_meter.update(X)

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (1,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        sub = self.sub_meter.get()
        dev = self.variance_meter.get()**0.5

        score = (X - sub) / (dev + 1e-10)

        return abs(score) if self.absolute else score
