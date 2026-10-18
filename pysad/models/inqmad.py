import numpy as np
from sklearn.kernel_approximation import RBFSampler

from pysad.core.base_model import BaseModel
from pysad.models.relative_entropy import _finite_float, _int_at_least


def _unit_interval(value, name):
    """Returns `value` as a Python float, or raises if it is not a real number in [0, 1]."""
    value = _finite_float(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")

    return value


class Inqmad(BaseModel):
    r"""Scores each instance by its negated density under a density matrix of random Fourier features that forgets old normal instances (Inqmad) :cite:`gallego2022inqmad`.

    Inqmad stands for incremental quantum measurement anomaly detection. Each instance is mapped to a state :math:`\psi` with random Fourier features, and its density is the quantum measurement :math:`\hat{f}(x) = \psi^\top \rho \psi` of the paper's Eq. 3 (without its normalization constant). The anomaly score is the negated density, so higher scores mean more anomalous instances. The model follows the paper's Algorithms 1 and 2:

    1. The first ``initial_size`` instances form the initial sample. Once it is complete, :math:`\rho` is the mean of their :math:`\psi \psi^\top` (Eq. 2) and the threshold :math:`\tau` is the ``beta`` quantile of their densities under it.
    2. Each later instance is scored against :math:`\rho_t` before it is fitted. When it is fitted and its density is at least :math:`\tau`, so that it is classified as normal, the update is :math:`\rho_{t+1} = (1 - \alpha) \rho_t + \alpha \psi \psi^\top` (Eq. 4). Instances below :math:`\tau` leave :math:`\rho` unchanged.

    While the initial sample fills up, an instance is scored by the density that the paper's estimate approximates, the mean of :math:`k(x, x_i)^2` over the instances :math:`x_i` of the sample so far, where :math:`k` is the Gaussian kernel of the random Fourier features. Before anything has been fitted, the density is 0 and the score is 0.0, the most anomalous possible score. The paper's Algorithm 1 applies the :math:`\alpha` update to the initial sample as well, but its Eq. 2 and the authors' code average it uniformly, which this implementation follows. The authors' code also scores :math:`\|\rho \psi\|^2 = \psi^\top \rho^2 \psi` instead of Eq. 3. Unlike the paper, the random Fourier features are fixed rather than adaptive.

    The kernel bandwidth ``gamma`` has to be tuned to the scale of the inputs: too large a value makes every instance about equally dense.

    Args:
        input_shape (int): Number of features. Must be an int >= 1 (a NumPy integer is accepted, but not a bool): `TypeError` is raised for other types and `ValueError` for values below 1. Every instance must be an array of shape (input_shape,) or (1, input_shape), or `ValueError` is raised before the model changes.
        dim_x (int): Random Fourier features dimension. Must be an int >= 1 (a NumPy integer is accepted, but not a bool): `TypeError` is raised for other types and `ValueError` for values below 1.
        gamma (float): Kernel bandwidth of the random Fourier features.
        initial_size (int): Number of instances in the initial sample, the paper's :math:`n`. Must be an int >= 1 (a NumPy integer is accepted, but not a bool): `TypeError` is raised for other types and `ValueError` for values below 1 (Default=256).
        alpha (float): Forgetting rate of the update, the weight of each new normal instance in :math:`\rho`. A larger value forgets old instances faster. Must be a real number between 0 and 1 (Default=0.01).
        beta (float): Expected proportion of anomalies, the quantile of the initial sample's densities that becomes the threshold :math:`\tau`. Must be a real number between 0 and 1 (Default=0.1).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the random Fourier features. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self, input_shape, dim_x, gamma, initial_size=256, alpha=0.01, beta=0.1, random_state=None
    ):
        input_shape = _int_at_least(input_shape, "input_shape", 1)
        dim_x = _int_at_least(dim_x, "dim_x", 1)
        self.initial_size = _int_at_least(initial_size, "initial_size", 1)
        self.alpha = _unit_interval(alpha, "alpha")
        self.beta = _unit_interval(beta, "beta")
        self.gamma = gamma
        sampler = RBFSampler(gamma=gamma, n_components=dim_x, random_state=random_state)
        sampler.fit(np.zeros((1, input_shape)))
        self.weights = sampler.random_weights_
        self.offset = sampler.random_offset_
        # The initial sample, until it is complete; then rho and tau replace it.
        self.initial = np.empty((0, input_shape))
        self.rho = None
        self.tau = None

    def _instance(self, X):
        """``X`` as a (1, num_features) row, or raises if it is not one instance of the fitted feature count."""
        num_features = self.weights.shape[0]
        X = np.asarray(X)
        if X.shape not in ((num_features,), (1, num_features)):
            raise ValueError(
                f"Inqmad expects one instance of shape ({num_features},) or (1, {num_features}), got an array of shape {X.shape}."
            )

        return X.reshape(1, num_features)

    def _states(self, X):
        """The unit-norm random Fourier feature states of the rows of ``X``."""
        vals = np.cos(np.atleast_2d(X) @ self.weights + self.offset)
        return vals / np.linalg.norm(vals, axis=1, keepdims=True)

    def _measure(self, X):
        """The state of the row ``X`` (None before rho exists) and its density."""
        if self.rho is None:
            if len(self.initial) == 0:
                return None, 0.0
            sq_dists = np.sum((self.initial - X) ** 2, axis=1)
            return None, np.mean(np.exp(-2 * self.gamma * sq_dists))

        psi = self._states(X)[0]
        return psi, psi @ self.rho @ psi

    def _fit(self, X, psi, density):
        if self.rho is None:
            self.initial = np.vstack([self.initial, X])
            if len(self.initial) == self.initial_size:
                states = self._states(self.initial)
                self.rho = states.T @ states / len(states)
                densities = np.einsum("ni,ij,nj->n", states, self.rho, states)
                self.tau = np.quantile(densities, self.beta)
                self.initial = None
        elif density >= self.tau:
            self.rho *= 1 - self.alpha
            self.rho += self.alpha * np.outer(psi, psi)

    def fit_partial(self, X, y=None):
        """Updates the model with the next instance: adds it to the initial sample, or, once the sample is complete, adds it to the density matrix if its density is at least the threshold.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.

        Raises:
            ValueError: If `X` is not of shape (input_shape,) or (1, input_shape). The model is left unchanged.
        """
        X = self._instance(X)
        self._fit(X, *self._measure(X))

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The negative of the estimated density for the input instance, as a Python float. Higher scores (lower estimated density) represent more anomalous instances. Before anything has been fitted, the density is 0 and the score is 0.0, the most anomalous possible score.

        Raises:
            ValueError: If `X` is not of shape (input_shape,) or (1, input_shape).
        """
        return -self._measure(self._instance(X))[1]

    def fit_score_partial(self, X, y=None):
        r"""Scores the next instance against the density matrix :math:`\rho_t` (or the initial sample) before it, and then fits it, as in the paper's Algorithm 2. The instance's own state therefore does not add to its density.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance, as in `score_partial`, so 0.0 for the first instance.

        Raises:
            ValueError: If `X` is not of shape (input_shape,) or (1, input_shape). The model is left unchanged.
        """
        X = self._instance(X)
        psi, density = self._measure(X)
        self._fit(X, psi, density)

        return -density
