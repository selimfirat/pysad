import numpy as np
from pysad.core.base_transformer import BaseTransformer


def _murmurhash3_x86_32(seeds, byte_matrix):
    """Vectorized 32-bit MurmurHash3 (x86_32), matching ``mmh3.hash(s, signed=False, seed=k)``.

    Computes the hash of every row of `byte_matrix` (all of the same byte length) against every
    seed in `seeds` in one shot, instead of calling into `mmh3` once per (seed, key) pair.

    Args:
        seeds (np.uint32 array of shape (n_seeds,)): Seeds to hash with.
        byte_matrix (np.uint8 array of shape (n_keys, length)): Raw bytes of each (equal-length) key.

    Returns:
        np.uint32 array of shape (n_seeds, n_keys): Unsigned hash of each (seed, key) pair.
    """
    c1 = np.uint32(0xcc9e2d51)
    c2 = np.uint32(0x1b873593)
    n_keys, length = byte_matrix.shape
    nblocks = length // 4

    # h1 starts at the seed and is broadcast across all keys.
    h1 = np.tile(seeds.astype(np.uint32)[:, np.newaxis], (1, n_keys))

    for block in range(nblocks):
        b = byte_matrix[:, 4 * block:4 * block + 4].astype(np.uint32)
        k1 = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)).astype(np.uint32)
        k1 = (k1 * c1).astype(np.uint32)
        k1 = ((k1 << 15) | (k1 >> 17)).astype(np.uint32)
        k1 = (k1 * c2).astype(np.uint32)
        h1 = h1 ^ k1[np.newaxis, :]
        h1 = ((h1 << 13) | (h1 >> 19)).astype(np.uint32)
        h1 = (h1 * np.uint32(5) + np.uint32(0xe6546b64)).astype(np.uint32)

    tail_size = length & 3
    if tail_size:
        tail = byte_matrix[:, nblocks * 4:length].astype(np.uint32)
        k1 = np.zeros(n_keys, dtype=np.uint32)
        if tail_size >= 3:
            k1 = k1 ^ (tail[:, 2] << 16).astype(np.uint32)
        if tail_size >= 2:
            k1 = k1 ^ (tail[:, 1] << 8).astype(np.uint32)
        k1 = k1 ^ tail[:, 0]
        k1 = (k1 * c1).astype(np.uint32)
        k1 = ((k1 << 15) | (k1 >> 17)).astype(np.uint32)
        k1 = (k1 * c2).astype(np.uint32)
        h1 = h1 ^ k1[np.newaxis, :]

    h1 = h1 ^ np.uint32(length)
    h1 = h1 ^ (h1 >> 16)
    h1 = (h1 * np.uint32(0x85ebca6b)).astype(np.uint32)
    h1 = h1 ^ (h1 >> 13)
    h1 = (h1 * np.uint32(0xc2b2ae35)).astype(np.uint32)
    h1 = h1 ^ (h1 >> 16)

    return h1


class StreamhashProjector(BaseTransformer):
    """Streamhash projection method  from Manzoor et. al.that is similar (or equivalent) to SparseRandomProjection. :cite:`xstream` The implementation is taken from the `cmuxstream-core repository <https://github.com/cmuxstream/cmuxstream-core>`_.

        Args:
            num_components (int): The number of dimensions that the target will be projected into.
            density (float): Density parameter of the streamhash projection.
    """

    # Defaults for instances unpickled from older pysad versions, whose __dict__ predates
    # these attributes; __init__ below overrides them for freshly constructed instances.
    _R = None
    _R_ndim = None

    def __init__(self, num_components, density=1 / 3.0):
        super().__init__(num_components)
        self.keys = np.arange(0, num_components, 1)
        self.constant = np.sqrt(1. / density) / np.sqrt(num_components)
        self.density = density
        self.n_components = num_components
        self._R = None
        self._R_ndim = None

    def fit_partial(self, X):
        """Fits particular (next) timestep's features to train the projector.

        Args:
            X (np.float64 array of shape (n_components,)): Input feature vector.

        Returns:
            object: self.
        """
        return self

    def transform_partial(self, X):
        """Projects particular (next) timestep's vector to (possibly) lower dimensional space.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            projected_X (np.float64 array of shape (num_components,)): Projected feature vector.
        """
        X = X.reshape(1, -1)

        ndim = X.shape[1]

        R = self._get_projection_matrix(ndim)

        Y = np.dot(X, R.T).squeeze()

        return Y

    def _get_projection_matrix(self, ndim):
        """Returns the projection matrix for the given number of features, building and caching it on first use.

        The matrix only depends on `ndim`, `self.keys`, `self.density` and `self.constant`, so it is
        rebuilt only when a sample with a different `ndim` than the cached one arrives.

        Args:
            ndim (int): Number of features of the incoming sample.

        Returns:
            R (np.float64 array of shape (num_components, ndim)): Projection matrix.
        """
        if self._R is None or self._R_ndim != ndim:
            self._R = self._build_projection_matrix(ndim)
            self._R_ndim = ndim

        return self._R

    def _build_projection_matrix(self, ndim):
        """Builds the projection matrix by hashing the ASCII decimal string of every feature index.

        Feature indices are grouped by the byte length of their decimal representation (e.g. 0-9,
        10-99, ...), since MurmurHash3 processes fixed-length blocks; each group is then hashed for
        all `self.keys` seeds at once via `_murmurhash3_x86_32`, instead of hashing one entry at a
        time with `mmh3.hash`.

        Args:
            ndim (int): Number of features of the incoming sample.

        Returns:
            R (np.float64 array of shape (num_components, ndim)): Projection matrix.
        """
        seeds = self.keys.astype(np.uint32)
        density = self.density

        feature_indices = np.arange(ndim)
        feature_strings = [str(i) for i in feature_indices]
        lengths = np.array([len(s) for s in feature_strings])

        R = np.empty((self.n_components, ndim), dtype=np.float64)

        for length in np.unique(lengths):
            group = np.flatnonzero(lengths == length)
            byte_matrix = np.array(
                [[ord(c) for c in feature_strings[i]] for i in group],
                dtype=np.uint8,
            )

            hashes = _murmurhash3_x86_32(seeds, byte_matrix)
            hash_values = hashes.astype(np.float64) / (2.0 ** 32 - 1)

            R[:, group] = np.where(
                hash_values <= density / 2.0, -1 * self.constant,
                np.where(hash_values <= density, self.constant, 0.0),
            )

        return R
