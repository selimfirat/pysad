import numpy as np

from pysad.core.base_transformer import BaseTransformer


def _murmurhash3_x86_32(seeds: np.ndarray, byte_matrix: np.ndarray) -> np.ndarray:
    """Vectorized 32-bit MurmurHash3 (x86_32), matching ``mmh3.hash(s, signed=False, seed=k)``.

    Computes the hash of every row of `byte_matrix` (all of the same byte length) against every
    seed in `seeds` in one shot, instead of calling into `mmh3` once per (seed, key) pair.

    Args:
        seeds (np.uint32 array of shape (n_seeds,)): Seeds to hash with.
        byte_matrix (np.uint8 array of shape (n_keys, length)): Raw bytes of each (equal-length) key.

    Returns:
        np.uint32 array of shape (n_seeds, n_keys): Unsigned hash of each (seed, key) pair.
    """
    c1 = np.uint32(0xCC9E2D51)
    c2 = np.uint32(0x1B873593)
    n_keys, length = byte_matrix.shape
    nblocks = length // 4

    # h1 starts at the seed and is broadcast across all keys.
    h1 = np.tile(seeds.astype(np.uint32)[:, np.newaxis], (1, n_keys))

    for block in range(nblocks):
        b = byte_matrix[:, 4 * block : 4 * block + 4].astype(np.uint32)
        k1 = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)).astype(np.uint32)
        k1 = (k1 * c1).astype(np.uint32)
        k1 = ((k1 << 15) | (k1 >> 17)).astype(np.uint32)
        k1 = (k1 * c2).astype(np.uint32)
        h1 = h1 ^ k1[np.newaxis, :]
        h1 = ((h1 << 13) | (h1 >> 19)).astype(np.uint32)
        h1 = (h1 * np.uint32(5) + np.uint32(0xE6546B64)).astype(np.uint32)

    tail_size = length & 3
    if tail_size:
        tail = byte_matrix[:, nblocks * 4 : length].astype(np.uint32)
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
    h1 = (h1 * np.uint32(0x85EBCA6B)).astype(np.uint32)
    h1 = h1 ^ (h1 >> 13)
    h1 = (h1 * np.uint32(0xC2B2AE35)).astype(np.uint32)
    h1 = h1 ^ (h1 >> 16)

    return h1


class StreamhashProjector(BaseTransformer):
    """Projects each feature vector with the sparse StreamHash random projection.

    Args:
        num_components (int): The number of dimensions that the target will be projected into.
        density (float): Fraction of non-zero entries in the projection matrix (Default=1/3).
    """

    _R: np.ndarray | None = None
    _R_ndim: int | None = None

    def __init__(self, num_components: int, density: float = 1 / 3.0) -> None:
        super().__init__(num_components)
        self.keys: np.ndarray = np.arange(0, num_components, 1)
        self.constant: float = float(np.sqrt(1.0 / density) / np.sqrt(num_components))
        self.density: float = density
        self.n_components: int = num_components
        self._R: np.ndarray | None = None
        self._R_ndim: int | None = None

    def fit_partial(self, X: np.ndarray) -> "StreamhashProjector":
        """Fits particular (next) timestep's features to train the projector.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            object: self.
        """
        return self

    def transform_partial(self, X: np.ndarray) -> np.ndarray:
        """Projects particular (next) timestep's vector to lower dimensional space.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (num_components,): Projected feature vector.
        """
        X = X.reshape(1, -1)

        ndim = X.shape[1]

        R = self._get_projection_matrix(ndim)

        Y = np.dot(X, R.T).squeeze()

        return Y

    def _get_projection_matrix(self, ndim: int) -> np.ndarray:
        """Returns the projection matrix for the given number of features.

        Args:
            ndim (int): Number of features of the incoming sample.

        Returns:
            R (np.float64 array of shape (num_components, ndim)): Projection matrix.
        """
        if self._R is None or self._R_ndim != ndim:
            self._R = self._build_projection_matrix(ndim)
            self._R_ndim = ndim

        return self._R

    def _build_projection_matrix(self, ndim: int) -> np.ndarray:
        """Builds the projection matrix by hashing feature indices.

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
                [[ord(c) for c in feature_strings[i]] for i in group], dtype=np.uint8
            )

            hashes = _murmurhash3_x86_32(seeds, byte_matrix)
            hash_values = hashes.astype(np.float64) / (2.0**32 - 1)

            R[:, group] = np.where(
                hash_values <= density / 2.0,
                -1 * self.constant,
                np.where(hash_values <= density, self.constant, 0.0),
            )

        return R
