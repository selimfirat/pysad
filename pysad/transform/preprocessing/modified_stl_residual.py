import numpy as np

from pysad.core.base_transformer import BaseTransformer
from pysad.utils import Window

# The STL functions below follow statsmodels 0.15.0 (statsmodels/tsa/stl/_stl.pyx,
# https://github.com/statsmodels/statsmodels), a port of the NETLIB Fortran STL of
# Cleveland et al. (1990), restricted to periodic STL with jumps of 1 and vectorized over the
# estimation points. For licensing information, see the end of this file.


def _loess(Y, J, xs, h, degree, rng, rw=None):
    # One LOESS estimate per row, as STL._est: Y and J (num_points, width) hold the values and
    # the 1-based positions of each window, xs and h (num_points,) the estimation position and
    # the half width. Entries outside a window have J = inf. Rows without weight give NaN.
    r = np.abs(J - xs[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(r <= 0.001 * h[:, None], 1.0, (1.0 - (r / h[:, None]) ** 3) ** 3)
        w[r > 0.999 * h[:, None]] = 0.0
        if rw is not None:
            w *= rw
        total = w.sum(axis=1)
        w /= total[:, None]
        if degree > 0:
            J = np.where(np.isfinite(J), J, 0.0)
            center = (w * J).sum(axis=1)
            d = J - center[:, None]
            c = (w * d**2).sum(axis=1)
            linear = (h > 0) & (np.sqrt(c) > 0.001 * rng)
            slope = (xs - center) / c
            w = np.where(linear[:, None], w * (slope[:, None] * d + 1.0), w)
        ys = (w * Y).sum(axis=1)
    ys[total <= 0] = np.nan
    return ys


def _smooth(y, length, degree, rw=None):
    # LOESS of the whole series over windows of `length` points, as STL._ess with a jump of 1.
    n = y.shape[0]
    i = np.arange(n)
    width = min(length, n)
    nleft = np.clip(i + 1 - (length + 2) // 2, 0, n - width)
    idx = nleft[:, None] + np.arange(width)  # (n, width)
    xs = i + 1.0
    h = np.maximum(xs - (nleft + 1), nleft + width - xs) + max(length - n, 0) // 2
    ys = _loess(y[idx], idx + 1.0, xs, h, degree, n - 1.0, None if rw is None else rw[idx])
    return np.where(np.isnan(ys), y, ys)


def _moving_average(x, length):
    return np.convolve(x, np.full(length, 1.0 / length), mode="valid")


def _periodic_stl_seasonal(y, period, robust):
    # The seasonal component of STL(y, period, seasonal=10 * n + 1, seasonal_deg=0, robust).fit().
    n = y.shape[0]
    seasonal_length = 10 * n + 1
    trend_length = int(np.ceil(1.5 * period / (1 - 1.5 / seasonal_length)))
    trend_length += trend_length % 2 == 0
    low_pass_length = period + 1 + period % 2
    num_inner, num_outer = (2, 15) if robust else (5, 0)

    # Row j of the cycle-subseries holds y[j::period] in its first k[j] columns. The seasonal
    # smoother runs over the whole subseries at positions 0 to k[j] + 1, so it extends each
    # subseries by one period on both sides.
    num_cycles = -(-n // period)
    k = (n - 1 - np.arange(period)) // period + 1  # (period,)
    valid = np.arange(num_cycles) < k[:, None]  # (period, num_cycles)
    cycle_idx = np.where(valid, np.arange(period)[:, None] + period * np.arange(num_cycles), 0)
    xs = np.arange(num_cycles + 2.0)
    h = np.maximum(xs - 1, k[:, None] - xs) + (seasonal_length - k[:, None]) // 2
    shape = (period, num_cycles + 2, num_cycles)
    J = np.broadcast_to(np.where(valid, np.arange(1.0, num_cycles + 1), np.inf)[:, None], shape)
    J = J.reshape(-1, num_cycles)
    xs = np.broadcast_to(xs, h.shape).ravel()
    h = h.ravel()
    out_valid = np.arange(num_cycles + 2) <= k[:, None] + 1  # (period, num_cycles + 2)
    out_idx = (np.arange(period)[:, None] + period * np.arange(num_cycles + 2))[out_valid]
    rows = np.arange(period)

    trend = np.zeros(n)
    rw = None
    for outer in range(num_outer + 1):
        for _ in range(num_inner):
            Y = (y - trend)[cycle_idx]
            cycle_rw = None if rw is None else np.broadcast_to(rw[cycle_idx][:, None], shape)
            cycle = _loess(
                Y=np.broadcast_to(np.where(valid, Y, 0.0)[:, None], shape).reshape(-1, num_cycles),
                J=J,
                xs=xs,
                h=h,
                degree=0,
                rng=None,
                rw=None if cycle_rw is None else cycle_rw.reshape(-1, num_cycles),
            ).reshape(period, num_cycles + 2)
            # Without weight, a subseries value keeps its own value and an end copies its neighbor.
            inner = cycle[:, 1:-1]
            fallback = np.isnan(inner) & valid
            inner[fallback] = Y[fallback]
            cycle[:, 0] = np.where(np.isnan(cycle[:, 0]), cycle[:, 1], cycle[:, 0])
            end = cycle[rows, k + 1]
            cycle[rows, k + 1] = np.where(np.isnan(end), cycle[rows, k], end)
            extended = np.empty(n + 2 * period)
            extended[out_idx] = cycle[out_valid]

            low_pass = _moving_average(extended, period)
            low_pass = _moving_average(_moving_average(low_pass, period), 3)
            season = extended[period : period + n] - _smooth(low_pass, low_pass_length, 1)
            trend = _smooth(y - season, trend_length, 1, rw)
        if outer == num_outer:
            break
        # Bisquare robustness weights with 6 times the median absolute residual, as STL._rwts.
        residual = np.abs(y - (trend + season))
        mid = np.partition(residual, [n // 2, n - n // 2 - 1])
        cmad = 3.0 * (mid[n // 2] + mid[n - n // 2 - 1])
        if cmad == 0:
            rw = np.ones(n)
            continue
        rw = np.where(residual <= 0.999 * cmad, (1.0 - (residual / cmad) ** 2) ** 2, 0.0)
        rw[residual <= 0.001 * cmad] = 1.0
    return season


class ModifiedSTLResidualTransformer(BaseTransformer):
    """Removes the seasonal component and the median from a univariate series, as S-ESD and S-H-ESD do :cite:`hochenbaum2017automatic`.

    The transformer follows the residual construction of the paper. It estimates the seasonal component with periodic STL, replaces the trend component of STL with the median of the raw series, and returns ``X - seasonal - median(X)``. Periodic STL is what the reference implementation of the paper uses (``stl(s.window = "periodic")`` in R): the seasonal smoother is a constant (degree 0) LOESS with a span of ``10 * n + 1`` for ``n`` observations, so the seasonal component repeats exactly in every period and a few anomalous periods cannot bend it. It does only the residual step of the paper. Use :class:`pysad.models.SeasonalESD` or :class:`pysad.models.SeasonalHybridESD` for the full detector.

    The partial transforms return ``[0.0]`` until the window holds at least ``2 * period`` observations, because STL needs two full periods.

    Args:
        period (int): Number of observations in one seasonal period.
        window_size (int or None): Number of recent observations used for partial transforms. If None, only ``transform_window`` works, and the partial methods raise ``ValueError`` (Default=None).
        robust (bool): Whether to use robust STL fitting (Default=True).

    Raises:
        ValueError: If ``period`` is less than 2, or if ``window_size`` is not None and less than ``2 * period``.
    """

    def __init__(self, period, window_size=None, robust=True):
        super().__init__(-1)

        if period < 2:
            raise ValueError("period must be greater than 1.")

        if window_size is not None and window_size < 2 * period:
            raise ValueError("window_size must be at least 2 * period.")

        self.period = period
        self.window_size = window_size
        self.robust = robust
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
        X = self._as_univariate(X)

        if X.shape[0] < 2 * self.period:
            raise ValueError("At least 2 * period observations are required for STL.")

        seasonal = _periodic_stl_seasonal(X, self.period, self.robust)
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
            return np.zeros(1)

        return self.transform_window(values)[-1:]

    def transform_partial(self, X):
        """Returns the residual of the next value in the window that ends with it, without adding it to the window.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.

        Returns:
            np.float64 array of shape (1,): The modified STL residual of ``X``, or ``[0.0]`` if the window has fewer than ``2 * period`` observations.
        """
        if self.window is None:
            raise ValueError("window_size is required for partial transforms.")

        return self._latest_residual(self._candidate_values(X))

    def fit_transform_partial(self, X):
        """Adds the next value to the window and returns its residual.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.

        Returns:
            np.float64 array of shape (1,): The modified STL residual of ``X``, or ``[0.0]`` if the window has fewer than ``2 * period`` observations.
        """
        self.fit_partial(X)
        return self._latest_residual(self._current_values())


# The STL functions above are adapted from statsmodels 0.15.0, which is distributed under the
# following license:
#
# Copyright (C) 2006, Jonathan E. Taylor
# All rights reserved.
#
# Copyright (c) 2006-2008 Scipy Developers.
# All rights reserved.
#
# Copyright (c) 2009-2018 statsmodels Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of statsmodels nor the names of its contributors
#      may be used to endorse or promote products derived from this software
#      without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL STATSMODELS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.
