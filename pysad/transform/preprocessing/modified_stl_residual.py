import numpy as np

from pysad.core.base_transformer import BaseTransformer
from pysad.utils import Window


class ModifiedSTLResidualTransformer(BaseTransformer):
    """Modified STL residual transformer used by S-ESD and S-H-ESD.

    The transformer follows the residual construction in
    :cite:`hochenbaum2017automatic`: estimate the seasonal component with STL,
    replace STL's trend component with the median of the raw series, and return
    ``X - seasonal - median(X)``. It owns only the paper's residual step; use
    :class:`pysad.models.SeasonalESD` or
    :class:`pysad.models.SeasonalHybridESD` for the full detector.

    Args:
        period (int): Number of observations in one seasonal period.
        window_size (int): Number of recent observations used for partial
            transforms.
        robust (bool): Whether to use robust STL fitting. (Default=True).
        **stl_kwargs: Additional keyword arguments passed to
            ``statsmodels.tsa.seasonal.STL``.
    """

    def __init__(self, period, window_size=None, robust=True, **stl_kwargs):
        super().__init__(-1)

        if period < 2:
            raise ValueError("period must be greater than 1.")

        if window_size is not None and window_size < 2 * period:
            raise ValueError("window_size must be at least 2 * period.")

        self.period = period
        self.window_size = window_size
        self.robust = robust
        self.stl_kwargs = stl_kwargs
        self.window = Window(window_size) if window_size is not None else None

    def _normalize_input(self, X):
        if isinstance(X, tuple):
            X = X[0]

        return X

    def _as_univariate(self, X):
        X = self._normalize_input(X)
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 2 and X.shape[1] == 1:
            X = X.ravel()
        elif X.ndim != 1:
            raise ValueError("ModifiedSTLResidualTransformer supports univariate inputs.")

        return X

    def transform_window(self, X):
        """Transforms a full window into modified STL residuals.

        Args:
            X (np.float64 array of shape (num_instances,) or
                (num_instances, 1)): Window of univariate values.

        Returns:
            residual_X (np.float64 array of shape (num_instances,)): Modified STL residuals.
        """
        from statsmodels.tsa.seasonal import STL

        X = self._as_univariate(X)

        if X.shape[0] < 2 * self.period:
            raise ValueError("At least 2 * period observations are required for STL.")

        seasonal = STL(
            X,
            period=self.period,
            robust=self.robust,
            **self.stl_kwargs
        ).fit().seasonal
        return X - seasonal - np.median(X)

    def fit_partial(self, X):
        """Fits the next timestep by adding it to the residual window."""
        if self.window is None:
            raise ValueError("window_size is required for partial fitting.")

        X = self._as_univariate(X)
        if X.shape[0] != 1:
            raise ValueError("Partial fitting requires a single univariate value.")

        self.window.update(X[0])
        return self

    def _as_partial_value(self, X):
        X = self._as_univariate(X)
        if X.shape[0] != 1:
            raise ValueError("Partial operations require a single univariate value.")

        return X[0]

    def _current_values(self):
        return np.asarray(self.window.get(), dtype=np.float64)

    def _candidate_values(self, X):
        value = self._as_partial_value(X)
        values = self.window.get() + [value]
        values = values[-self.window_size:]
        return np.asarray(values, dtype=np.float64)

    def _latest_residual(self, values):
        residuals = self.transform_window(values)
        return residuals[-1]

    def transform_partial(self, X):
        """Returns the latest residual for the next candidate window."""
        if self.window is None:
            raise ValueError("window_size is required for partial transforms.")

        return self._latest_residual(self._candidate_values(X))

    def fit_transform_partial(self, X):
        """Adds and transforms the next timestep without adding it twice."""
        self.fit_partial(X)
        return self._latest_residual(self._current_values())
