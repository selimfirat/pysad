import numpy as np

from pysad.core.base_transformer import BaseTransformer
from pysad.utils import Window


class ModifiedSTLResidualTransformer(BaseTransformer):
    """Removes the seasonal component and the median from a univariate series, as S-ESD and S-H-ESD do :cite:`hochenbaum2017automatic`.

    The transformer follows the residual construction of the paper. It estimates the seasonal component with STL, replaces the trend component of STL with the median of the raw series, and returns ``X - seasonal - median(X)``. It does only the residual step of the paper. Use :class:`pysad.models.SeasonalESD` or :class:`pysad.models.SeasonalHybridESD` for the full detector.

    The partial transforms return 0.0 until the window holds at least ``2 * period`` observations, because STL needs two full periods.

    Args:
        period (int): Number of observations in one seasonal period.
        window_size (int or None): Number of recent observations used for partial transforms. If None, only ``transform_window`` works, and the partial methods raise ``ValueError`` (Default=None).
        robust (bool): Whether to use robust STL fitting (Default=True).
        **stl_kwargs: Additional keyword arguments passed to ``statsmodels.tsa.seasonal.STL``.

    Raises:
        ValueError: If ``period`` is less than 2, or if ``window_size`` is not None and less than ``2 * period``.
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

    def _as_univariate(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 2 and X.shape[1] == 1:
            X = X.ravel()
        elif X.ndim != 1:
            raise ValueError("ModifiedSTLResidualTransformer supports univariate inputs.")

        return X

    def transform_window(self, X):
        """Transforms a full window into modified STL residuals.

        Args:
            X (np.float64 array of shape (num_instances,) or (num_instances, 1)): Window of univariate values.

        Returns:
            np.float64 array of shape (num_instances,): Modified STL residuals.

        Raises:
            ValueError: If the window has fewer than ``2 * period`` observations.
        """
        from statsmodels.tsa.seasonal import STL

        X = self._as_univariate(X)

        if X.shape[0] < 2 * self.period:
            raise ValueError("At least 2 * period observations are required for STL.")

        seasonal = STL(X, period=self.period, robust=self.robust, **self.stl_kwargs).fit().seasonal
        return X - seasonal - np.median(X)

    def fit_partial(self, X):
        """Fits the next timestep by adding it to the residual window.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.

        Returns:
            object: self.
        """
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
        values = values[-self.window_size :]
        return np.asarray(values, dtype=np.float64)

    def _latest_residual(self, values):
        if values.shape[0] < 2 * self.period:
            return 0.0

        residuals = self.transform_window(values)
        return residuals[-1]

    def transform_partial(self, X):
        """Returns the residual of the next value in the window that ends with it, without adding it to the window.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.

        Returns:
            float: The modified STL residual of ``X``, or 0.0 if the window has fewer than ``2 * period`` observations.
        """
        if self.window is None:
            raise ValueError("window_size is required for partial transforms.")

        return self._latest_residual(self._candidate_values(X))

    def fit_transform_partial(self, X):
        """Adds the next value to the window and returns its residual.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.

        Returns:
            float: The modified STL residual of ``X``, or 0.0 if the window has fewer than ``2 * period`` observations.
        """
        self.fit_partial(X)
        return self._latest_residual(self._current_values())
