from collections import deque

import numpy as np

from pysad.core.base_model import BaseModel


class MedianAbsoluteDeviation(BaseModel):
    """Scores each value of a univariate stream by its distance to the median of a sliding window, in units of the scaled median absolute deviation (MAD) of that window :cite:`hochenbaum2017automatic`.

    The model keeps the last ``window_size`` values. For each score it computes the median of the window and its MAD, ``median_i |x_i - median_j(x_j)|`` (Eq. 7 of the paper), and returns ``(x - median) / (b * MAD)``. ``fit_score_partial`` adds the value to the window before it scores the value, as the paper's test does on its batch.

    This is a streaming MAD score component, not the S-H-ESD method of the paper. It does not apply STL decomposition or generalized ESD. Use :class:`pysad.models.SeasonalHybridESD` for the modified STL plus median and MAD ESD detector of the paper.

    Args:
        window_size (int): Number of recent values to compute the median and the MAD on. It must be at least 1 (Default=500).
        absolute (bool): Whether to output score's absolute value (Default=True).
        b (float): Scale factor of the MAD. The default value ``1.4826`` is used for normally distributed data. See :cite:`hochenbaum2017automatic` for details (Default=1.4826).

    Raises:
        ValueError: If ``window_size`` is less than 1.
    """

    def __init__(self, window_size=500, absolute=True, b=1.4826):
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")

        self.window_size = window_size
        self.b = b
        self.absolute = absolute
        self.window = deque(maxlen=window_size)

    def fit_partial(self, X, y=None):
        """Updates the model with the next instance.

        Args:
            X (np.float64 array of shape (1,)): The instance to fit. This model is univariate.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        if len(X) != 1:
            raise ValueError("MedianAbsoluteDeviation supports univariate inputs.")

        self.window.append(float(X[0]))

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (1,)): The instance to score. This model is univariate.

        Returns:
            float: The anomalousness score of the input instance.

        Raises:
            ValueError: If the model has not been fitted on any value yet.
        """
        if not self.window:
            raise ValueError("Call fit_partial before score_partial.")

        window = np.asarray(self.window)
        median = np.median(window)
        mad = self.b * np.median(np.abs(window - median))
        score = (X - median) / (mad + 1e-10)

        return abs(score) if self.absolute else score
