from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import norm, solve
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class FRAHST(BaseModel):
    r"""Fast rank-adaptive row-Householder subspace tracking :cite:`teixeira2010data`.

    FRAHST tracks an orthonormal basis of the principal subspace of uncentered
    input. Figure 1 updates this basis with a Householder reflection. The rank
    starts at one. Two exponentially weighted energies track the input and its
    projection onto the previous basis. If the captured fraction falls below
    ``energy_low``, the model adds the normalized residual of the updated basis.
    If the fraction exceeds ``energy_high``, it removes the last basis vector.

    The score is the binary alarm from Figure 2. A rank increase scores one,
    except on the first instance or directly after another rank increase.
    Other instances score zero. ``score_partial`` computes the next transition
    without learning. ``fit_partial`` applies it. These alarms detect changes
    in correlation, rather than isolated large values within the learned subspace.
    Use a scaler or shingling before this model when the stream needs them.

    ``basis_`` holds the learned basis, with shape ``(num_features, rank_)``.
    ``rank_`` stays between one and the number of features.
    For n features and rank r, memory costs :math:`O(nr + r^2)`.
    Each transition costs :math:`O(nr + r^3)`, independent of stream length.

    Implementation choices and deviations from the paper:

    - This class uses the direct solve in Figure 1. It omits the optional QR updates in Section 3.4, so the rank cost is cubic.
    - Initialization draws one normal vector and normalizes it. ``random_state`` controls this draw. The initial S-matrix uses sigma = 1e-4.
    - Zero inputs leave the model unchanged, as in Section 3.3. For nonzero inputs with zero residual, this class skips only the reflection. It still applies equations 13c and 13k-13u. Skipping the entire transition as Section 3.3 suggests locks the model at full rank.
    - The reflection threshold is sqrt(machine epsilon) times max(1, input energy). Rank growth uses the same threshold. A second projection removes rounding error from the new direction before normalization.
    - If rounding leaves no residual direction after the basis update, the model skips rank growth. It does not append an arbitrary direction.
    - The paper reports alarms after fitting. This class predicts the same alarm before fitting to follow the PySAD score API. Before initialization, it returns zero.

    The unlicensed ``MrKriss/Old-PhD-Code`` reference also continues rank updates
    without a reflection. This class follows Figure 1 instead of its extra
    previous-reflection term. It uses the unnormalized residual energy in S,
    random initialization, and Figure 2 alarm suppression. The reference uses
    a normalized residual in S and offers identity initialization.

    Args:
        alpha (float): Forgetting factor for both energies and the subspace update (Default=0.96).
        energy_low (float): Lower captured-energy fraction that triggers rank growth (Default=0.96).
        energy_high (float): Upper captured-energy fraction that triggers rank reduction (Default=0.98).
        random_state (int, np.random.RandomState or None): Seed or generator for the initial basis. None uses NumPy's global generator (Default=None).
    """

    def __init__(
        self,
        alpha: float = 0.96,
        energy_low: float = 0.96,
        energy_high: float = 0.98,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between zero and one.")
        if not 0.0 < energy_low < energy_high < 1.0:
            raise ValueError("Energy fractions must satisfy 0 < energy_low < energy_high < 1.")
        self.alpha = alpha
        self.energy_low = energy_low
        self.energy_high = energy_high
        self.random_state = random_state
        self.rng_ = check_random_state(random_state)
        self.basis_: NDArray[np.float64] | None = None
        self.s_ = np.array([[1e-4]])
        self.rank_ = 1
        self.energy_ = 0.0
        self.captured_energy_ = 0.0
        self.count_ = 0
        self.last_increase_ = 0

    def _step(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float, bool]:
        """Computes Figure 1 without changing the current state."""
        assert self.basis_ is not None
        Q = self.basis_
        h = Q.T.dot(X)
        energy = float(X.dot(X))
        captured = float(h.dot(h))
        residual_energy = energy - captured
        tolerance = np.sqrt(np.finfo(float).eps) * max(1.0, energy)
        if energy == 0.0:
            return Q, self.s_, self.energy_, self.captured_energy_, False

        S = self.alpha * self.s_ + np.outer(h, h)
        if residual_energy > tolerance:
            Q, S = self._reflect(X, Q, h, S, residual_energy)
        energy = self.alpha * self.energy_ + energy
        captured = self.alpha * self.captured_energy_ + captured

        increase = False
        if captured < self.energy_low * energy and self.rank_ < X.size:
            residual = X - Q.dot(Q.T.dot(X))
            # A second projection removes rounding error before normalization.
            residual -= Q.dot(Q.T.dot(residual))
            length = norm(residual)
            if length * length > tolerance:
                Q = np.column_stack((Q, residual / length))
                expanded = np.zeros((self.rank_ + 1, self.rank_ + 1))
                expanded[:-1, :-1] = S
                expanded[-1, -1] = length * length
                S = expanded
                increase = True
        elif captured > self.energy_high * energy and self.rank_ > 1:
            Q = Q[:, :-1].copy()
            S = S[:-1, :-1].copy()
        return Q, S, energy, captured, increase

    @staticmethod
    def _reflect(
        X: NDArray[np.float64],
        Q: NDArray[np.float64],
        h: NDArray[np.float64],
        S: NDArray[np.float64],
        residual_energy: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Applies the Householder reflection in equations 13d-13j."""
        root = np.sqrt(residual_energy)
        b = solve(S.T, root * h, check_finite=False)
        # Equations 13e-13j. hypot avoids squaring a large solve result.
        length = np.hypot(1.0, norm(b))
        phi = np.sqrt(0.5 + 0.5 / length)
        delta = phi / root
        v = -(b / length) / (2.0 * phi)
        S -= np.outer(v / delta, h)
        e = delta * X - Q.dot(delta * h - v)
        Q = Q - 2.0 * np.outer(e, v)
        return Q, S

    def fit_partial(self, X: NDArray[np.float64], y: int | None = None) -> FRAHST:
        """Updates the subspace and its rank with one instance.

        Args:
            X (np.ndarray): Input with shape (num_features,).
            y (int or None): Unused label (Default=None).

        Returns:
            FRAHST: The updated model.
        """
        if self.basis_ is None:
            basis = self.rng_.normal(size=(X.size, 1))
            self.basis_ = basis / norm(basis)
        self.basis_, self.s_, self.energy_, self.captured_energy_, increase = self._step(X)
        self.rank_ = self.basis_.shape[1]
        self.count_ += 1
        if increase:
            self.last_increase_ = self.count_
        return self

    def score_partial(self, X: NDArray[np.float64]) -> float:
        """Returns the next rank-increase alarm without learning.

        Args:
            X (np.ndarray): Input with shape (num_features,).

        Returns:
            float: One for an unsuppressed rank increase, otherwise zero.
        """
        if self.basis_ is None or self.count_ + 1 <= self.last_increase_ + 1:
            return 0.0
        return float(self._step(X)[-1])
