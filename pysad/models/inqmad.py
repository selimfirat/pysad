import numpy as np
from sklearn.kernel_approximation import RBFSampler

from pysad.core.base_model import BaseModel


class Inqmad(BaseModel):
    r"""The Inqmad (incremental quantum measurement anomaly detection) model for row-streaming data :cite:`gallego2022inqmad`. Each instance is mapped to a state :math:`\psi` with random Fourier features, the model keeps a density matrix :math:`\rho` that averages :math:`\psi \psi^\top` over every fitted instance, and the anomaly score is the negated density estimate :math:`-\psi^\top \rho \psi` of the paper's Eq. 3 (without its normalization constant), so higher scores mean more anomalous instances. As in the paper, which measures the density against :math:`\rho_t` before updating it, `fit_score_partial` scores each instance before fitting it, so the instance's own state does not add to its density; before anything has been fitted, :math:`\rho` is zero and the score is 0.0 (density 0), the most anomalous possible score. Unlike the paper, the random Fourier features are fixed rather than adaptive, there is no :math:`\tau` threshold, every fitted instance updates :math:`\rho` rather than only those classified as normal, and :math:`\rho` is a uniform running average rather than the paper's :math:`\alpha`-forgetting update.

    The kernel bandwidth ``gamma`` has to be tuned to the scale of the inputs: too large a value makes every instance about equally dense.

    Args:
        input_shape (int): number of features
        dim_x (int): random Fourier features dimension
        gamma (float): kernel parameter for the random Fourier features
        random_state (int, np.random.RandomState or None): Seed or random number generator for the random Fourier features. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(self, input_shape, dim_x, gamma, random_state=None):
        sampler = RBFSampler(gamma=gamma, n_components=dim_x, random_state=random_state)
        sampler.fit(np.zeros((1, input_shape)))
        self.weights = sampler.random_weights_
        self.offset = sampler.random_offset_
        # Sum of psi psi^T over the fitted instances, kept in float64 so that long streams do not lose updates.
        self.rho = np.zeros((dim_x, dim_x))
        self.num_fitted = 0

    def _states(self, X):
        """The unit-norm random Fourier feature states of the rows of ``X``."""
        vals = np.cos(np.atleast_2d(X) @ self.weights + self.offset)
        return vals / np.linalg.norm(vals, axis=1, keepdims=True)

    def _fit_states(self, psi):
        self.rho += psi.T @ psi
        self.num_fitted += psi.shape[0]

    def _score_states(self, psi):
        if self.num_fitted == 0:
            # rho is still zero, so every density is 0.
            return np.zeros(psi.shape[0])

        # One score per row, so that BaseModel rejects several rows.
        return -np.einsum("ni,ij,nj->n", psi, self.rho, psi) / self.num_fitted

    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: Returns the self.
        """
        self._fit_states(self._states(X))

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            score (float): The negative of the estimated density for the input instance, as a Python float. Higher scores (lower estimated density) represent more anomalous instances. Before anything has been fitted, the density is 0 and the score is 0.0, the most anomalous possible score.
        """
        return self._score_states(self._states(X))

    def fit_score_partial(self, X, y=None):
        r"""Scores the next instance against the density matrix of the instances fitted before it, and then fits it, as the paper measures the density against :math:`\rho_t` before updating it. The instance's own state therefore does not add to its density.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance, as in `score_partial`, so 0.0 for the first instance.
        """
        psi = self._states(X)
        score = self._score_states(psi)
        self._fit_states(psi)

        return score
