import numpy as np
from pysad.core.base_model import BaseModel
from pysad.transform.projection.streamhash_projector import StreamhashProjector
from pysad.utils import get_minmax_array

# Try to import JAX dependencies, otherwise define a flag to indicate they're missing
try:
    import warnings
    # Suppress numpy.core deprecation warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="numpy.core is deprecated")
        import jax
        from jax import jit
        import jax.numpy as jnp
    JAX_AVAILABLE = True
except (ImportError, AttributeError):
    # Handle both missing JAX and JAX-NumPy compatibility issues
    JAX_AVAILABLE = False

from functools import partial
from sklearn.kernel_approximation import RBFSampler



class Inqmad(BaseModel):
    """The Inqmad model for row-streaming data :cite:`xstream`. (a) an initial normal stream data point is captured (b) those points are mapped to a Hilbert space using adaptive Fourier features (AFF) (c) a memory density matrix $\rho_t$ is initialized using the points from the last step and the $\tau$-threshold value is defined (d) the stream of data points arrives (e) each point is mapped to a Hilbert space using (AFF) (f) a quantum measurement is performed between the streaming point and the memory density matrix $\rho_t$ (g) a $\tau$-threshold value is used to classify normal and anomalous points (h) detect whether the point was classified as normal (i) compute the updated memory density matrix $\rho_{t+1}$ using the normal classified streaming point (j) update the memory density matrix $\rho_t$ with the new matrix $\rho_{t+1}.

    Args:
        input_shape (int): number of features
        dim_x (int): random Fourier features dimension 
        gamma (int): kernel parameter for the random Fourier features 
        random_state (int): initial random state for the random Fourier features
        batch_size (int): training samples processed by iteration

    Note:
        This model requires JAX and JAXlib (version 0.6.1 or higher) to be installed. You can install them using:
        `pip install jax>=0.6.1 jaxlib>=0.6.1`
        or via the optional dependency:
        `pip install pysad[inqmad]`
        
        When using NumPy 2.0 or higher, JAX 0.6.1+ is required for compatibility.
    """

    def __init__(
            self,
            input_shape, 
            dim_x, 
            gamma, 
            random_state=42, 
            batch_size = 300
            ):
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

        if(X.ndim == 1):
            X = np.expand_dims(X, axis=0)

        self.inqmad.initial_train(jnp.array(X), 1)
        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            score (float): The anomalousness score of the input instance.
        """
        if(X.ndim == 1):
            X = np.expand_dims(X, axis=0)

        return self.inqmad.predict(X)



class QFeatureMap_rff():
  """The random Fourier features for Inqmad :cite:`inqmad`.

    Args:
        input_shape (int): number of features
        dim (int): random Fourier features dimension 
        gamma (int): kernel parameter for the random Fourier features 
        random_state (int): initial random state for the random Fourier features
    """

  def __init__(
          self,
          input_dim: int,
          dim: int = 100,
          gamma: float = 1,
          random_state=None,
          **kwargs
  ):
      super().__init__(**kwargs)
      self.input_dim = input_dim
      self.dim = dim
      self.gamma = gamma
      self.random_state = random_state
      self.vmap_compute = jax.jit(jax.vmap(self.compute, in_axes=(0, None, None, None), out_axes=0))

  
  def build(self):
    rbf_sampler = RBFSampler(
            gamma=self.gamma,
            n_components=self.dim,
            random_state=self.random_state)
    x = np.zeros(shape=(1, self.input_dim))
    rbf_sampler.fit(x)

    self.rbf_sampler = rbf_sampler
    self.weights = jnp.array(rbf_sampler.random_weights_)
    self.offset = jnp.array(rbf_sampler.random_offset_)
    self.dim = rbf_sampler.get_params()['n_components']

  def update_rff(self, weights, offset):
    self.weights = jnp.array(weights)
    self.offset = jnp.array(offset)


  def get_dim(self, num_features):
    return self.dim

  @staticmethod
  def compute(X, weights, offset, dim):
    vals = jnp.dot(X, weights) + offset
    #vals = jnp.einsum('i,ik->k', X, weights) + offset
    vals = jnp.cos(vals)
    vals *= jnp.sqrt(2.) / jnp.sqrt(dim)
    return vals
    
  @partial(jit, static_argnums=(0,))
  def __call__(self, X):
    vals = self.vmap_compute(X, self.weights, self.offset, self.dim)
    norms = jnp.linalg.norm(vals, axis=1)
    psi = vals / norms[:, jnp.newaxis]
    return psi   


class InqMeasurement():
  """The Inqmad model for row-streaming data :cite:`xstream`. (a) an initial normal stream data point is captured (b) those points are mapped to a Hilbert space using adaptive Fourier features (AFF) (c) a memory density matrix $\rho_t$ is initialized using the points from the last step and the $\tau$-threshold value is defined (d) the stream of data points arrives (e) each point is mapped to a Hilbert space using (AFF) (f) a quantum measurement is performed between the streaming point and the memory density matrix $\rho_t$ (g) a $\tau$-threshold value is used to classify normal and anomalous points (h) detect whether the point was classified as normal (i) compute the updated memory density matrix $\rho_{t+1}$ using the normal classified streaming point (j) update the memory density matrix $\rho_t$ with the new matrix $\rho_{t+1}.

    Args:
        input_shape (int): number of features
        dim_x (int): random Fourier features dimension 
        gamma (int): kernel parammeter for the random Fourier features 
        random_state (int): initial random state for the random Fourier features
        batch_size (int): training samples processed by iteration
    """



  def __init__(self, input_shape, dim_x, gamma, random_state=42, batch_size = 300):
    self.gamma = gamma
    self.dim_x = dim_x
    self.fm_x = QFeatureMap_rff( input_dim=input_shape, dim = dim_x, gamma = gamma, random_state = random_state)
    self.fm_x.build()
    self.num_samples = 0 
    self.train_pure_batch = jax.jit(jax.vmap(self.train_pure, in_axes=(0)))
    self.collapse_batch = jax.jit(jax.vmap(self.collapse, in_axes=(0, None)))
    self.sum_batch = jax.jit(self.sum)
    self.batch_size = batch_size

  @staticmethod
  def train_pure(inputs):
    oper = jnp.einsum(
        '...i,...j->...ij',
        inputs, jnp.conj(inputs),
        optimize='optimal') # shape (b, nx, nx)
    return oper

  @staticmethod
  def sum(rho_res):
    return jnp.sum(rho_res, axis=0) 


  @staticmethod
  @partial(jit, static_argnums=(1,2,3,4))
  def compute_training_jit(batch, alpha, fm_x, train_pure_batch, sum_batch, rho):
      inputs = fm_x(batch)
      rho_res = train_pure_batch(inputs)
      rho_res = sum_batch(rho_res)
      return jnp.add((alpha)*rho_res, (1-alpha)*rho) if rho is not None else rho_res

  @staticmethod
  def compute_training(values, alpha, perm, i, batch_size, fm_x, train_pure_batch, sum_batch, rho, compute_training_jit):
      batch_idx = perm[i * batch_size: (i + 1)*batch_size]
      batch = values[batch_idx, :]
      return compute_training_jit(batch, alpha, fm_x, train_pure_batch, sum_batch, rho)

  def initial_train(self, values, alpha):
    num_batches = InqMeasurement.obtain_params_batches(values, self.batch_size)
    num_train = values.shape[0]
    perm = jnp.arange(num_train)
    for i in range(num_batches):
      if hasattr(self, "rho_res"):
        self.rho_res = self.compute_training(values, alpha, perm, i,
                            self.batch_size, self.fm_x, 
                            self.train_pure_batch, self.sum_batch, 
                            self.rho_res, self.compute_training_jit)
      else:
        self.rho_res = self.compute_training(values, alpha, perm, i, 
                            self.batch_size, self.fm_x,
                            self.train_pure_batch, self.sum_batch, None, 
                            self.compute_training_jit)
    self.num_samples += values.shape[0]  

    

  @staticmethod
  def collapse(inputs, rho_res):
    rho_h = jnp.matmul(jnp.conj(inputs), rho_res)
    rho_res = jnp.einsum(
        '...i, ...i -> ...',
        rho_h, jnp.conj(rho_h), 
        optimize='optimal') # shape (b,)
    return rho_res

  @staticmethod
  def obtain_params_batches(values, batch_size):
    num_train = values.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return num_batches

  

  @partial(jit, static_argnums=(0,))
  def predict(self, values):
    num_batches = InqMeasurement.obtain_params_batches(values, self.batch_size)
    results = None
    rho_res = self.rho_res / self.num_samples
    num_train = values.shape[0]
    perm = jnp.arange(num_train)
    for i in range(num_batches):
      batch_idx = perm[i * self.batch_size: (i + 1)*self.batch_size]
      batch = values[batch_idx, :]

      inputs = self.fm_x(batch)
      batch_probs = self.collapse_batch(inputs, rho_res)
      results = jnp.concatenate([results, batch_probs], axis=0) if results is not None else batch_probs
    return results

