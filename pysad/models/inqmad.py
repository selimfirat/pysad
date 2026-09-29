import numpy as np

from pysad.core.base_model import BaseModel

# Try to import JAX dependencies, otherwise define a flag to indicate they're missing
try:
    import warnings

    # Suppress numpy.core deprecation warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="numpy.core is deprecated")
        import jax
        import jax.numpy as jnp
        from jax import jit
    JAX_AVAILABLE = True
except (ImportError, AttributeError):
    # Handle both missing JAX and JAX-NumPy compatibility issues.
    # Stub ``jit`` so class-body ``@partial(jit, ...)`` decorators still
    # resolve at import time; ``Inqmad.__init__`` raises ImportError with
    # the install hint when JAX is actually missing (see #174).
    JAX_AVAILABLE = False

    def jit(fun=None, **_kwargs):
        if fun is None:
            return lambda f: f
        return fun

    jnp = None

from functools import partial

from sklearn.kernel_approximation import RBFSampler


class Inqmad(BaseModel):
    r"""The Inqmad (incremental quantum measurement anomaly detection) model for row-streaming data :cite:`gallego2022inqmad`. Each instance is mapped to a state :math:`\psi` with random Fourier features, the model keeps a density matrix :math:`\rho` that averages :math:`\psi \psi^\top` over every fitted instance, and the anomaly score is the negated density estimate :math:`-\psi^\top \rho \psi` of the paper's Eq. 3 (without its normalization constant), so higher scores mean more anomalous instances. As in the paper, which measures the density against :math:`\rho_t` before updating it, `fit_score_partial` scores each instance before fitting it, so the instance's own state does not add to its density; before anything has been fitted, :math:`\rho` is zero and the score is 0.0 (density 0), the most anomalous possible score. Unlike the paper, the random Fourier features are fixed rather than adaptive, there is no :math:`\tau` threshold, every fitted instance updates :math:`\rho` rather than only those classified as normal, and :math:`\rho` is a uniform running average rather than the paper's :math:`\alpha`-forgetting update.

    Args:
        input_shape (int): number of features
        dim_x (int): random Fourier features dimension
        gamma (float): kernel parameter for the random Fourier features
        random_state (int, np.random.RandomState or None): Seed or random number generator for the random Fourier features. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
        batch_size (int): training samples processed by iteration

    Note:
        This model requires JAX and JAXlib (version 0.6.1 or higher) to be installed. You can install them using:
        `pip install jax>=0.6.1 jaxlib>=0.6.1`
        or via the optional dependency:
        `pip install pysad[inqmad]`

        When using NumPy 2.0 or higher, JAX 0.6.1+ is required for compatibility.
    """

    def __init__(self, input_shape, dim_x, gamma, random_state=None, batch_size=300):
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX dependencies are required to use the Inqmad model. "
                "Please install jax and jaxlib via pip: "
                "`pip install jax jaxlib` or `pip install pysad[inqmad]`"
            )
        self.inqmad = InqMeasurement(input_shape, dim_x, gamma, random_state, batch_size)

    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: Returns the self.
        """

        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)

        self.inqmad.initial_train(jnp.array(X))
        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            score (float): The negative of the estimated density for the input instance, as a Python float. Higher scores (lower estimated density) represent more anomalous instances. Before anything has been fitted, the density is 0 and the score is 0.0, the most anomalous possible score.
        """
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)

        if self.inqmad.num_samples == 0:
            # rho is still zero, so every density is 0. One score per row, so that BaseModel
            # still rejects several rows.
            return np.zeros(X.shape[0])

        # Negate on the host: a unary minus on the jax Array would dispatch another device op per call.
        # The array is returned as is, so BaseModel converts it to a float and rejects several rows.
        return -np.asarray(self.inqmad.predict(X))

    def fit_score_partial(self, X, y=None):
        r"""Scores the next instance against the density matrix of the instances fitted before it, and then fits it, as the paper measures the density against :math:`\rho_t` before updating it. The instance's own state therefore does not add to its density.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance, as in `score_partial`, so 0.0 for the first instance.
        """
        score = self.score_partial(X)
        self.fit_partial(X, y)

        return score


class QFeatureMap_rff:
    """The random Fourier features for Inqmad :cite:`gallego2022inqmad`.

    Args:
        input_dim (int): number of features
        dim (int): random Fourier features dimension
        gamma (float): kernel parameter for the random Fourier features
        random_state (int, np.random.RandomState or None): Seed or random number generator for the random Fourier features. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self, input_dim: int, dim: int = 100, gamma: float = 1, random_state=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.dim = dim
        self.gamma = gamma
        self.random_state = random_state
        self.vmap_compute = jax.jit(
            jax.vmap(self.compute, in_axes=(0, None, None, None), out_axes=0)
        )

    def build(self):
        rbf_sampler = RBFSampler(
            gamma=self.gamma, n_components=self.dim, random_state=self.random_state
        )
        x = np.zeros(shape=(1, self.input_dim))
        rbf_sampler.fit(x)

        self.rbf_sampler = rbf_sampler
        self.weights = jnp.array(rbf_sampler.random_weights_)
        self.offset = jnp.array(rbf_sampler.random_offset_)
        self.dim = rbf_sampler.get_params()["n_components"]

    def update_rff(self, weights, offset):
        self.weights = jnp.array(weights)
        self.offset = jnp.array(offset)

    def get_dim(self, num_features):
        return self.dim

    @staticmethod
    def compute(X, weights, offset, dim):
        vals = jnp.dot(X, weights) + offset
        # vals = jnp.einsum('i,ik->k', X, weights) + offset
        vals = jnp.cos(vals)
        vals *= jnp.sqrt(2.0) / jnp.sqrt(dim)
        return vals

    @partial(jit, static_argnums=(0,))
    def __call__(self, X):
        vals = self.vmap_compute(X, self.weights, self.offset, self.dim)
        norms = jnp.linalg.norm(vals, axis=1)
        psi = vals / norms[:, jnp.newaxis]
        return psi


class InqMeasurement:
    r"""The density matrix estimator behind :class:`Inqmad` :cite:`gallego2022inqmad`. It sums the outer products :math:`\psi \psi^\top` of the random Fourier feature states of the fitted instances and estimates the density of a query state :math:`\psi` as :math:`\psi^\top \rho \psi`, where :math:`\rho` is that sum divided by the number of fitted instances.

    Args:
        input_shape (int): number of features
        dim_x (int): random Fourier features dimension
        gamma (float): kernel parameter for the random Fourier features
        random_state (int, np.random.RandomState or None): Seed or random number generator for the random Fourier features. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
        batch_size (int): training samples processed by iteration
    """

    def __init__(self, input_shape, dim_x, gamma, random_state=None, batch_size=300):
        self.gamma = gamma
        self.dim_x = dim_x
        self.fm_x = QFeatureMap_rff(
            input_dim=input_shape, dim=dim_x, gamma=gamma, random_state=random_state
        )
        self.fm_x.build()
        self.num_samples = 0
        self.collapse_batch = jax.jit(jax.vmap(self.collapse, in_axes=(0, None)))
        self.batch_size = batch_size

    @staticmethod
    def train_pure(inputs):
        oper = jnp.einsum(
            "...i,...j->...ij", inputs, jnp.conj(inputs), optimize="optimal"
        )  # shape (b, nx, nx)
        return oper

    @staticmethod
    def sum(rho_res):
        return jnp.sum(rho_res, axis=0)

    @staticmethod
    @partial(jit, static_argnums=(1,))
    def compute_training_jit(batch, fm_x, rho):
        inputs = fm_x(batch)
        rho_res = jax.vmap(InqMeasurement.train_pure)(inputs)
        rho_res = InqMeasurement.sum(rho_res)
        # Sum, do not replace: predict divides by num_samples, so rho is the
        # uniform average over every fitted instance.
        return jnp.add(rho_res, rho) if rho is not None else rho_res

    def initial_train(self, values):
        num_batches = InqMeasurement.obtain_params_batches(values, self.batch_size)
        for i in range(num_batches):
            batch = values[i * self.batch_size : (i + 1) * self.batch_size, :]
            self.rho_res = self.compute_training_jit(
                batch, self.fm_x, getattr(self, "rho_res", None)
            )
        self.num_samples += values.shape[0]

    @staticmethod
    def collapse(inputs, rho_res):
        # Density estimate of the paper's Eq. 3, psi^T rho psi.
        return jnp.einsum(
            "...i, ij, ...j -> ...", jnp.conj(inputs), rho_res, inputs, optimize="optimal"
        )

    @staticmethod
    def obtain_params_batches(values, batch_size):
        num_train = values.shape[0]
        num_complete_batches, leftover = divmod(num_train, batch_size)
        num_batches = num_complete_batches + bool(leftover)
        return num_batches

    @staticmethod
    @partial(jit, static_argnums=(3, 4, 5))
    def predict_pure(values, rho_res, num_samples, fm_x, collapse_batch, batch_size):
        num_batches = InqMeasurement.obtain_params_batches(values, batch_size)
        results = None
        # num_samples must stay a traced (non-static) argument: it changes with
        # every fit, so marking it static would recompile on every score.
        rho_res = rho_res / num_samples
        num_train = values.shape[0]
        perm = jnp.arange(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            batch = values[batch_idx, :]

            inputs = fm_x(batch)
            batch_probs = collapse_batch(inputs, rho_res)
            results = (
                jnp.concatenate([results, batch_probs], axis=0)
                if results is not None
                else batch_probs
            )
        return results

    def predict(self, values):
        # rho_res and num_samples are passed as traced arguments (not read from
        # self inside the jitted function) so a later fit_partial's update is
        # picked up instead of being baked into a stale compiled trace. The count
        # goes in as a float, which jit cannot overflow the way it does an int32.
        return self.predict_pure(
            values,
            self.rho_res,
            float(self.num_samples),
            self.fm_x,
            self.collapse_batch,
            self.batch_size,
        )
