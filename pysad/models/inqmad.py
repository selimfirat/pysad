import numpy as np
from sklearn.kernel_approximation import RBFSampler

from pysad.core.base_model import BaseModel
from pysad.models.relative_entropy import _int_at_least


class Inqmad(BaseModel):
    r"""The Inqmad (incremental quantum measurement anomaly detection) model for row-streaming data :cite:`gallego2022inqmad`. Each instance is mapped to a state :math:`\psi` with random Fourier features, the model keeps a density matrix :math:`\rho` that averages :math:`\psi \psi^\top` over every fitted instance, and the anomaly score is the negated density estimate :math:`-\psi^\top \rho \psi` of the paper's Eq. 3 (without its normalization constant), so higher scores mean more anomalous instances. As in the paper, which measures the density against :math:`\rho_t` before updating it, `fit_score_partial` scores each instance before fitting it, so the instance's own state does not add to its density; before anything has been fitted, :math:`\rho` is zero and the score is 0.0 (density 0), the most anomalous possible score. Unlike the paper, the random Fourier features are fixed rather than adaptive, there is no :math:`\tau` threshold, every fitted instance updates :math:`\rho` rather than only those classified as normal, and :math:`\rho` is a uniform running average rather than the paper's :math:`\alpha`-forgetting update.

    The kernel bandwidth ``gamma`` has to be tuned to the scale of the inputs: too large a value makes every instance about equally dense.

    Args:
        input_shape (int): Number of features. Must be an int >= 1 (a NumPy integer is accepted, but not a bool): `TypeError` is raised for other types and `ValueError` for values below 1. Every instance must be an array of shape (input_shape,) or (1, input_shape), or `ValueError` is raised before the model changes.
        dim_x (int): Random Fourier features dimension. Must be an int >= 1 (a NumPy integer is accepted, but not a bool): `TypeError` is raised for other types and `ValueError` for values below 1.
        gamma (float): kernel parameter for the random Fourier features
        random_state (int, np.random.RandomState or None): Seed or random number generator for the random Fourier features. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(self, input_shape, dim_x, gamma, random_state=None):
        input_shape = _int_at_least(input_shape, "input_shape", 1)
        dim_x = _int_at_least(dim_x, "dim_x", 1)
        sampler = RBFSampler(gamma=gamma, n_components=dim_x, random_state=random_state)
        sampler.fit(np.zeros((1, input_shape)))
        self.weights = sampler.random_weights_
        self.offset = sampler.random_offset_
        # Sum of psi psi^T over the fitted instances, kept in float64 so that long streams do not lose updates.
        self.rho = np.zeros((dim_x, dim_x))
        self.num_fitted = 0

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

        Raises:
            ValueError: If `X` is not of shape (input_shape,) or (1, input_shape). The model is left unchanged.
        """
        self._fit_states(self._states(self._instance(X)))

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            score (float): The negative of the estimated density for the input instance, as a Python float. Higher scores (lower estimated density) represent more anomalous instances. Before anything has been fitted, the density is 0 and the score is 0.0, the most anomalous possible score.

        Raises:
            ValueError: If `X` is not of shape (input_shape,) or (1, input_shape).
        """
        return self._score_states(self._states(self._instance(X)))

    def fit_score_partial(self, X, y=None):
        r"""Scores the next instance against the density matrix of the instances fitted before it, and then fits it, as the paper measures the density against :math:`\rho_t` before updating it. The instance's own state therefore does not add to its density.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance, as in `score_partial`, so 0.0 for the first instance.

        Raises:
            ValueError: If `X` is not of shape (input_shape,) or (1, input_shape). The model is left unchanged.
        """
        psi = self._states(self._instance(X))
        score = self._score_states(psi)
        self._fit_states(psi)

        return score
