from __future__ import annotations

import numpy as np

from pysad.core.base_transformer import BaseTransformer
from pysad.statistics.basic import VarianceMeter
from pysad.transform.preprocessing.modified_stl_residual import _periodic_stl_seasonal
from pysad.utils.validation import check_univariate

# The first and second differences of the trend that end at the last three points a, b and c,
# over the trends (tau_{a-2}, tau_{a-1}, tau_a, tau_b, tau_c).
_FIRST_DIFFERENCES = np.array([[0, -1, 1, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, -1, 1]], dtype=float)
_SECOND_DIFFERENCES = np.array([[1, -2, 1, 0, 0], [0, 1, -2, 1, 0], [0, 0, 1, -2, 1]], dtype=float)
# The residual and seasonal terms of one point, over its (trend, seasonal) pair.
_DATA_BLOCK = np.array([[1.0, 1.0], [1.0, 2.0]])
# Added to the absolute differences in the IRLS weights, so that a flat trend does not divide by 0.
_EPS = 1e-5


class OneShotSTL(BaseTransformer):
    """Decomposes a univariate series online into trend, seasonal and residual with OneShotSTL, and returns the residual :cite:`he2023oneshotstl`.

    OneShotSTL solves the JointSTL problem of the paper (Eq. 7) online. At each point, it fits the trend :math:`\\tau` and the seasonal component :math:`s` of all points since the initialization by minimizing

    .. math::

        \\sum_j (\\tau_j + s_j - y_j)^2 + \\sum_j (s_j - v_{j \\bmod T})^2 + \\lambda \\sum_j |\\tau_j - \\tau_{j-1}| + \\lambda \\sum_j |\\tau_j - 2\\tau_{j-1} + \\tau_{j-2}|,

    where :math:`v` holds the seasonal component of the last period. It solves the problem with ``num_iterations`` iteratively reweighted least squares (IRLS) iterations, and keeps one online symmetric Doolittle (LDL) factorization of the banded linear system for each iteration (Algorithms 4 and 5). Each point updates only the last rows of the factorizations, so a point costs the same at any stream length. The transformer returns the residual ``X - trend - seasonal`` of the newest point, and then sets :math:`v_{t \\bmod T}` to its seasonal component.

    The first ``num_init_periods * period`` points initialize :math:`v` with the seasonal component of robust periodic STL, as the reference implementation of the paper does. The partial transforms return ``[0.0]`` for these points.

    If ``shift_window`` is above 0, the transformer handles seasonality shifts as in Section 3.4. It keeps the running mean and standard deviation of the residuals against the unshifted :math:`v` (Algorithm 6). If the residual of a point lies more than ``shift_threshold`` standard deviations from the mean, it also decomposes the point against :math:`v` shifted by each :math:`\\Delta \\in [-H, H]`, and keeps the shift with the smallest absolute residual.

    Deviations from the paper:

    * The paper sets :math:`\\lambda` by matching OneShotSTL to batch STL on the training data. Here ``lam`` is a fixed parameter.
    * The IRLS weights :math:`1 / (2|d|)` of Eq. 4 and 5 use :math:`1 / (2(|d| + 10^{-5}))`, as the reference implementation does, so that a flat trend does not divide by zero.

    Args:
        period (int): Number of observations in one seasonal period, :math:`T`.
        lam (float): Weight :math:`\\lambda` of the first and second difference penalties of the trend. A larger value gives a smoother trend (Default=1.0).
        num_init_periods (int): Number of periods that initialize the seasonal component. Must be at least 3 (Default=4).
        num_iterations (int): Number of IRLS iterations for each point, :math:`I` (Default=8).
        shift_window (int): Largest seasonality shift :math:`H` that the transformer tries. 0 turns off the shift handling. The paper uses 20 in its experiments. The search also lowers the residual of a spike, because a shifted seasonal value can lie closer to it, so the default turns it off (Default=0).
        shift_threshold (float): Number of standard deviations of the residual above which the transformer tries the shifts, :math:`n` (Default=5.0).

    Raises:
        ValueError: If ``period`` is less than 2 or ``num_init_periods`` is less than 3.
    """

    def __init__(
        self,
        period: int,
        lam: float = 1.0,
        num_init_periods: int = 4,
        num_iterations: int = 8,
        shift_window: int = 0,
        shift_threshold: float = 5.0,
    ) -> None:
        super().__init__(1)

        if period < 2:
            raise ValueError("period must be greater than 1.")
        if num_init_periods < 3:
            raise ValueError("num_init_periods must be at least 3.")

        self.period = period
        self.lam = lam
        self.num_init_periods = num_init_periods
        self.num_iterations = num_iterations
        self.shift_window = shift_window
        self.shift_threshold = shift_threshold

        self.init_values_: list[float] | None = []
        self.seasonal_ = np.zeros(period)
        self.num_online_ = 0
        self.residual_meter_ = VarianceMeter()
        # For each IRLS iteration: the rows of L (8, 4), D (4,) and the forward substituted
        # right-hand side (4,) of the two points before the last two (Algorithm 4).
        self.L_ = np.zeros((num_iterations, 8, 4))
        self.D_ = np.ones((num_iterations, 4))
        self.z_ = np.zeros((num_iterations, 4))
        # The IRLS weights p and q of the last two points, for each iteration.
        self.p_ = np.zeros((num_iterations, 2))
        self.q_ = np.zeros((num_iterations, 2))
        # The right-hand side (y, y + v) of the last two points.
        self.rhs_ = np.zeros((2, 2))

    def _decompose(self, y: float, v: float) -> tuple[float, float, tuple[np.ndarray, ...]]:
        # Solves the system of every IRLS iteration for the newest point (Algorithm 5), and
        # returns its trend and seasonal component and the updated state.
        num_online = self.num_online_
        A = np.zeros((6, 6))  # over (tau, s) of the last three points a, b and c
        for k in range(3):
            if num_online + k >= 2:
                A[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = _DATA_BLOCK
            else:  # no point yet: a decoupled unit block keeps the system regular
                A[2 * k, 2 * k] = A[2 * k + 1, 2 * k + 1] = 1.0
        b = np.concatenate([self.rhs_.ravel(), [y, y + v]])
        L = np.zeros((self.num_iterations, 8, 4))
        D = np.empty((self.num_iterations, 4))
        z = np.empty((self.num_iterations, 4))
        p = np.empty((self.num_iterations, 2))
        q = np.empty((self.num_iterations, 2))
        p_new = q_new = 1.0
        for i in range(self.num_iterations):
            p_c = p_new if num_online >= 1 else 0.0
            q_c = q_new if num_online >= 2 else 0.0
            p_i = np.append(self.p_[i], p_c)
            q_i = np.append(self.q_[i], q_c)
            # (5, 5) penalty over (tau_{a-2}, tau_{a-1}, tau_a, tau_b, tau_c).
            penalty = (_FIRST_DIFFERENCES.T * p_i) @ _FIRST_DIFFERENCES
            penalty += (_SECOND_DIFFERENCES.T * q_i) @ _SECOND_DIFFERENCES
            A_i = A.copy()
            A_i[::2, ::2] += self.lam * penalty[2:, 2:]
            tau, s, L_i, D_i, z_i = _online_doolittle(A_i, b, self.L_[i], self.D_[i], self.z_[i])
            L[i], D[i], z[i] = L_i, D_i, z_i
            p[i], q[i] = p_i[1:], q_i[1:]
            # Eq. 4 and 5 weights of the newest point for the next iteration.
            p_new = 0.5 / (abs(tau[2] - tau[1]) + _EPS)
            q_new = 0.5 / (abs(tau[2] - 2.0 * tau[1] + tau[0]) + _EPS)
        return tau[2], s, (L, D, z, p, q, b[2:].reshape(2, 2))

    def _step(self, X: np.ndarray, update: bool) -> np.ndarray:
        y = check_univariate(X, "OneShotSTL")
        if self.init_values_ is not None:
            if update:
                self._initialize(y)
            return np.zeros(1)

        phase = self.num_online_ % self.period
        trend, seasonal, state = self._decompose(y, self.seasonal_[phase])
        residual = unshifted = y - trend - seasonal
        meter = self.residual_meter_
        if (
            self.shift_window > 0
            and meter.count > 0
            and abs(unshifted - meter.mean) > self.shift_threshold * np.sqrt(meter.get())
        ):
            for shift in range(-self.shift_window, self.shift_window + 1):
                if shift == 0:
                    continue
                v = self.seasonal_[(phase + shift) % self.period]
                shifted = self._decompose(y, v)
                if abs(y - shifted[0] - shifted[1]) < abs(residual):
                    trend, seasonal, state = shifted
                    residual = y - trend - seasonal

        if update:
            self.L_, self.D_, self.z_, self.p_, self.q_, self.rhs_ = state
            self.seasonal_[phase] = seasonal
            meter.update(unshifted)
            self.num_online_ += 1
        return np.array([residual])

    def _initialize(self, y: float) -> None:
        assert self.init_values_ is not None
        self.init_values_.append(y)
        if len(self.init_values_) == self.num_init_periods * self.period:
            values = np.array(self.init_values_)
            # The STL seasonal component is periodic, and the series starts at phase 0.
            self.seasonal_ = _periodic_stl_seasonal(values, self.period, robust=True)[: self.period]
            self.init_values_ = None

    def fit_partial(self, X: np.ndarray) -> OneShotSTL:
        """Decomposes the next value and updates the decomposition with it.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.

        Returns:
            object: self.
        """
        self._step(X, update=True)
        return self

    def transform_partial(self, X: np.ndarray) -> np.ndarray:
        """Returns the residual of the next value, without updating the decomposition.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.

        Returns:
            np.float64 array of shape (1,): The residual of ``X``, or ``[0.0]`` during the initialization.
        """
        return self._step(X, update=False)

    def fit_transform_partial(self, X: np.ndarray) -> np.ndarray:
        """Decomposes the next value, updates the decomposition with it, and returns its residual.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.

        Returns:
            np.float64 array of shape (1,): The residual of ``X``, or ``[0.0]`` during the initialization.
        """
        return self._step(X, update=True)


def _online_doolittle(
    A: np.ndarray, b: np.ndarray, L_old: np.ndarray, D_old: np.ndarray, z_old: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    # Algorithm 4. The 10 unknowns are the (tau, s) pairs of the last five points. The first four
    # belong to points whose rows of the LDL factorization are final: L_old holds their columns
    # (8, 4), D_old and z_old their D and forward substituted right-hand side. A (6, 6) and b (6,)
    # hold the system rows of the last three points, which the newest point changes. The system
    # has a half bandwidth of 4. Returns the trends of the last three points, the seasonal of the
    # newest one, and the factorization state for the next point.
    L = np.zeros((10, 10))
    L[:8, :4] = L_old
    D = np.concatenate([D_old, np.empty(6)])
    z = np.concatenate([z_old, b])
    for k in range(4, 10):
        lk = L[k, k - 4 : k]
        dl = D[k - 4 : k] * lk
        D[k] = A[k - 4, k - 4] - dl @ lk
        z[k] -= lk @ z[k - 4 : k]
        L[k + 1 : k + 5, k] = (A[k - 3 : k + 1, k - 4] - L[k + 1 : k + 5, k - 4 : k] @ dl) / D[k]
    # Backward substitution is exact for the last unknowns, since it runs from the end.
    x = z[4:] / D[4:]
    for j in range(4, -1, -1):
        x[j] -= L[j + 5 : j + 9, j + 4] @ x[j + 1 : j + 5]
    return x[::2], float(x[5]), L[2:, 2:6], D[2:6], z[2:6]
