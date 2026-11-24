from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import norm

from pysad.core.base_model import BaseModel

__bibtex__ = r"""
@article{huang2015streaming,
  title={Streaming anomaly detection using randomized matrix sketching},
  author={Huang, Hao and Kasiviswanathan, Shiva Prasad},
  journal={Proceedings of the VLDB Endowment},
  volume={9},
  number={3},
  pages={192--203},
  year={2015},
  doi={10.14778/2850583.2850593}
}
"""


class ADeMS(BaseModel):
    """Scores each instance by how badly a Frequent Directions sketch of the earlier normal instances reconstructs it :cite:`huang2015streaming`.

    This is AnomDetect (Algorithm 1) with DetUpdate (Algorithm 3), called ADeMS in the paper. Each instance is one column, so each call to ``fit_partial`` is a timestep with batch size one. The model scales each instance to unit L2 norm before scoring and fitting. The score is the Euclidean reconstruction error ``||X - U U^T X||`` on the normalized input, using the basis from earlier accepted instances. ``score_partial`` never learns; ``fit_score_partial`` scores before learning.

    DetUpdate computes the SVD of ``[B, X]``, keeps ``sketch_size`` left singular vectors and shrinks their squared singular values by the smallest retained squared singular value. The scoring basis consists of the first ``num_components`` vectors BEFORE shrinkage, as specified in Algorithm 3. Memory is O(num_features * sketch_size); update time is O(num_features * sketch_size ** 2), independent of stream length.

    ``initial_X`` bootstraps the model from trusted normal instances (Section 4.1), in one DetUpdate call without threshold filtering. Later instances are accepted only when their score is at most ``threshold``. With no threshold, all instances are accepted. This corresponds to an infinite threshold in Algorithm 1. Without ``initial_X``, the sketch starts empty, and the initial score is 1 for non-zero instances. A finite threshold requires ``initial_X`` so it does not reject every new direction during initialization.

    Conventions where the paper leaves behavior unspecified: until the data has rank ``num_components``, only singular vectors with numerically nonzero singular values are used. Zero instances score zero and leave the sketch unchanged. Empty or all-zero training data gives an empty basis. The default sketch size is ceil(sqrt(num_features)), capped at num_features; the default component count is max(1, num_features // 5), capped at the sketch size. These combine the paper's suggested sketch size and experimental component count. Algorithm 3 permits equality of these counts, although its relative-error bound requires strictly fewer components than sketch columns. This deterministic variant has no random state.

    Compared with the MIT-licensed ``takuti/stream-anomaly-detect`` reference (``SketchUpdate``), this class follows the paper's normalization of every column. Scores agree on matching parameters. This class returns scores without learning, uses one instance per timestep, and omits the reference's optional batch percentile filtering. Rank-deficient initialization uses the numerical-rank convention above instead of arbitrary null-space directions.

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
        initial_X: NDArray[np.float64] | None = None,
    ) -> None:
        if threshold is not None and initial_X is None:
            raise ValueError(
                "threshold needs initial_X, because an empty basis scores each non-zero instance as 1."
            )
        self.num_components = num_components
        self.sketch_size = sketch_size
        self.threshold = threshold
        self.to_init = True
        if initial_X is not None:
            initial_X = np.asarray(initial_X, dtype=np.float64)
            self._init_model(initial_X.shape[1])
            if initial_X.shape[0] and np.any(initial_X):
                self._update(self._normalize_batch(initial_X.T))

    def _init_model(self, num_features: int) -> None:
        sketch_size = self.sketch_size or int(np.ceil(np.sqrt(num_features)))
        self.sketch_size_ = min(sketch_size, num_features)
        if self.num_components is not None and self.num_components > self.sketch_size_:
            raise ValueError(
                f"num_components must be at most the sketch size {self.sketch_size_}, got {self.num_components}."
            )
        default_components = min(max(1, num_features // 5), self.sketch_size_)
        self.num_components_ = self.num_components or default_components
        self.sketch_: NDArray[np.float64] = np.zeros((num_features, 0))
        self.components_: NDArray[np.float64] = np.zeros((num_features, 0))
        self.to_init = False

    @staticmethod
    def _normalize(x: NDArray[np.float64]) -> NDArray[np.float64] | None:
        """Normalize a single instance to unit L2 norm, with overflow/underflow protection."""
        scale = np.abs(x).max()
        if scale == 0.0:
            return None
        y = x / scale
        nrm = np.linalg.norm(y)
        if nrm == 0.0:
            return None
        unit: NDArray[np.float64] = y / nrm
        return unit

    def _normalize_batch(self, Y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalize each column of Y to unit L2 norm, dropping zero columns."""
        normalized = []
        for i in range(Y.shape[1]):
            col = self._normalize(Y[:, i])
            if col is not None:
                normalized.append(col)
        if not normalized:
            return np.zeros((Y.shape[0], 0))
        return np.column_stack(normalized)

    @staticmethod
    def _instance(X: NDArray[np.float64]) -> NDArray[np.float64]:
        # A fresh contiguous copy, so that every BLAS call sees the same memory layout and
        # alignment whatever array the caller passes, including a reused buffer.
        return np.array(X, dtype=np.float64, order="C", copy=True)

    def _update(self, Y: NDArray[np.float64]) -> None:
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

    def _residual(self, y: NDArray[np.float64]) -> float:
        U = self.components_
        return float(norm(y - U @ (U.T @ y)))

    def fit_partial(self, X: NDArray[np.float64], y: int | None = None) -> ADeMS:
        """Updates the sketch with the next instance, unless its score is above ``threshold``.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        if self.to_init:
            self._init_model(X.shape[0])
        x = self._instance(X)
        x_norm = self._normalize(x)
        if x_norm is None or (
            self.threshold is not None and self._residual(x_norm) > self.threshold
        ):
            return self
        self._update(x_norm[:, None])

        return self

    def score_partial(self, X: NDArray[np.float64]) -> float:
        """Scores the instance by the norm of its residual after the projection on the top singular vectors.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: The score of the instance, between 0 and 1. Higher scores represent more anomalous instances.
        """
        if self.to_init:
            # Before any fit, the basis is empty, so the residual is the norm of the normalized instance.
            x = self._instance(X)
            x_norm = self._normalize(x)
            return 0.0 if x_norm is None else 1.0
        x = self._instance(X)
        x_norm = self._normalize(x)
        return 0.0 if x_norm is None else self._residual(x_norm)
