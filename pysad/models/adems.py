from __future__ import annotations

import numpy as np
from scipy.linalg import norm

from pysad.core.base_model import BaseModel


class ADeMS(BaseModel):
    """Scores each instance by how badly a Frequent Directions sketch of the earlier normal instances reconstructs it :cite:`huang2015streaming`.

    This is AnomDetect (Algorithm 1) with DetUpdate (Algorithm 3), called ADeMS in the paper. Each instance is one column, so each call to ``fit_partial`` is a timestep with batch size one. The score is the Euclidean reconstruction error ``||X - U U^T X||`` on the original, uncentered input, using the basis from earlier accepted instances. No normalization is applied. ``score_partial`` never learns; ``fit_score_partial`` scores before learning.

    DetUpdate computes the SVD of ``[B, X]``, keeps ``sketch_size`` left singular vectors and shrinks their squared singular values by the smallest retained squared singular value. The scoring basis consists of the first ``num_components`` vectors BEFORE shrinkage, as specified in Algorithm 3. Memory is O(num_features * sketch_size); update time is O(num_features * sketch_size ** 2), independent of stream length.

    ``initial_X`` bootstraps the model from trusted normal instances (Section 4.1), in one DetUpdate call without threshold filtering. Later instances are accepted only when their score is at most ``threshold``. With no threshold, all instances are accepted. This corresponds to an infinite threshold in Algorithm 1. Without ``initial_X``, the sketch starts empty, and the initial score is the input norm. A finite threshold requires ``initial_X`` so it does not reject every new direction during initialization.

    Conventions where the paper leaves behavior unspecified: until the data has rank ``num_components``, only singular vectors with numerically nonzero singular values are used. Zero instances score zero and leave the sketch unchanged. Empty or all-zero training data gives an empty basis. The default sketch size is ceil(sqrt(num_features)), capped at num_features; the default component count is max(1, num_features // 5), capped at the sketch size. These combine the paper's suggested sketch size and experimental component count. Algorithm 3 permits equality of these counts, although its relative-error bound requires strictly fewer components than sketch columns. This deterministic variant has no random state.

    Compared with the MIT-licensed ``takuti/stream-anomaly-detect`` reference (``SketchUpdate``), this class follows the paper's raw-input equations instead of normalizing every column. Scores agree on unit-length, full-rank streams with matching parameters. This class returns scores without learning, uses one instance per timestep, and omits the reference's optional batch percentile filtering. Rank-deficient initialization uses the numerical-rank convention above instead of arbitrary null-space directions.

    Args:
        num_components (int or None): The number of singular vectors that reconstruct an instance, k in the paper. None uses num_features // 5, at least 1 and at most ``sketch_size`` (Default=None).
        sketch_size (int or None): The number of columns of the sketch, l in the paper. None uses ceil(sqrt(num_features)) (Default=None).
        threshold (float or None): The score above which ``fit_partial`` does not fit an instance, zeta in the paper. None fits every instance (Default=None).
        initial_X (np.float64 array of shape (num_instances, num_features) or None): Normal instances that build the first sketch without the threshold (Default=None).
    """

    def __init__(
        self,
        num_components: int | None = None,
        sketch_size: int | None = None,
        threshold: float | None = None,
        initial_X: np.ndarray | None = None,
    ) -> None:
        if threshold is not None and initial_X is None:
            raise ValueError(
                "threshold needs initial_X, because an empty basis scores each instance by its norm."
            )
        self.num_components = num_components
        self.sketch_size = sketch_size
        self.threshold = threshold
        self.to_init = True
        if initial_X is not None:
            initial_X = np.asarray(initial_X, dtype=np.float64)
            self._init_model(initial_X.shape[1])
            if initial_X.shape[0] and np.any(initial_X):
                self._update(initial_X.T)

    def _init_model(self, num_features: int) -> None:
        sketch_size = self.sketch_size or int(np.ceil(np.sqrt(num_features)))
        self.sketch_size_ = min(sketch_size, num_features)
        if self.num_components is not None and self.num_components > self.sketch_size_:
            raise ValueError(
                f"num_components must be at most the sketch size {self.sketch_size_}, got {self.num_components}."
            )
        default_components = min(max(1, num_features // 5), self.sketch_size_)
        self.num_components_ = self.num_components or default_components
        self.sketch_ = np.zeros((num_features, 0))
        self.components_ = np.zeros((num_features, 0))
        self.to_init = False

    def _update(self, Y: np.ndarray) -> None:
        # DetUpdate (Algorithm 3) with the new instances in the columns of Y, shape (num_features, n).
        D = np.hstack([self.sketch_, Y])  # (num_features, num_columns + n)
        U, s, _ = np.linalg.svd(D, full_matrices=False)
        ell = self.sketch_size_
        # Factoring out s avoids overflow/underflow from squaring singular values.
        retained = s[:ell]
        ratio = np.zeros_like(retained)
        if s.size >= ell:
            np.divide(s[ell - 1], retained, out=ratio, where=retained != 0)
        shrunk = retained * np.sqrt(np.maximum((1.0 - ratio) * (1.0 + ratio), 0.0))
        self.sketch_ = U[:, :ell] * shrunk
        # The singular vectors with nonzero singular values, as np.linalg.matrix_rank decides.
        k = min(
            self.num_components_, int((s > s[0] * (max(D.shape) * np.finfo(s.dtype).eps)).sum())
        )
        # A copy, since pickle stores a view as a contiguous array, which changes the rounding of _residual.
        self.components_ = U[:, :k].copy()

    def _residual(self, y: np.ndarray) -> float:
        U = self.components_
        return float(norm(y - U @ (U.T @ y)))

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> ADeMS:
        """Updates the sketch with the next instance, unless its score is above ``threshold``.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        if self.to_init:
            self._init_model(X.shape[0])
        if not np.any(X) or (self.threshold is not None and self._residual(X) > self.threshold):
            return self
        self._update(np.asarray(X, dtype=np.float64)[:, None])

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the instance by the norm of its residual after the projection on the top singular vectors.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: The Euclidean reconstruction error of the instance. Higher scores represent more anomalous instances.
        """
        if self.to_init:
            return float(norm(X))
        return self._residual(X)
