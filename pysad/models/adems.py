from __future__ import annotations

import numpy as np

from pysad.core.base_model import BaseModel


class ADeMS(BaseModel):
    """Scores each instance by how badly a Frequent Directions sketch of the earlier normal instances reconstructs it :cite:`huang2015streaming`.

    This is ADeMS, Algorithm AnomDetect of the paper with the deterministic sketch update DetUpdate (Algorithm 3). The model scales each instance to unit L2 norm. It keeps a sketch ``B`` of ``sketch_size`` columns, which approximates the normal instances seen so far, and the top ``num_components`` left singular vectors ``U`` of that sketch. The score of an instance ``y`` is the norm of its residual ``(I - U U^T) y``, a value between 0 and 1. To fit an instance, the model computes the singular value decomposition of ``[B, y]``, keeps the top ``sketch_size`` singular vectors, and subtracts the square of the smallest kept singular value from the square of each kept singular value. ``U`` is the top ``num_components`` left singular vectors of ``[B, y]``.

    If ``threshold`` is set, ``fit_partial`` fits only the instances whose score is at most ``threshold``, as the paper does, and skips the instances that it marks as anomalies. The paper builds the first ``U`` from a training set of normal instances without a threshold. ``initial_X`` is that training set. Since the model scores every instance 1 before it fits one, a ``threshold`` needs ``initial_X``.

    Deviations from the paper: the paper takes a batch of instances at each time step. This model takes one instance at a time, the batch size of 1 that the paper allows. Until the sketch has rank ``num_components``, ``U`` holds only the singular vectors with nonzero singular values. The model gives a zero instance a score of 0 and does not fit it, because it has no direction. The paper suggests ``sketch_size`` = sqrt(num_features) and sets ``num_components`` = num_features / 5 in its experiments, but also requires ``num_components`` <= ``sketch_size``. The defaults follow these rules: ``sketch_size`` is ceil(sqrt(num_features)) and ``num_components`` is num_features // 5, at least 1 and at most ``sketch_size``. ``sketch_size`` is at most num_features. The paper has no default for the threshold. By default the model fits every instance.

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
                "threshold needs initial_X, because the model scores every instance 1 before it fits one."
            )
        self.num_components = num_components
        self.sketch_size = sketch_size
        self.threshold = threshold
        self.to_init = True
        if initial_X is not None:
            initial_X = np.asarray(initial_X, dtype=np.float64)
            self._init_model(initial_X.shape[1])
            Y = [y for y in map(self._normalize, initial_X) if y is not None]
            self._update(np.column_stack(Y))

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

    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray | None:
        # Scales by the largest magnitude first, so that the norm neither overflows nor underflows.
        scale = np.abs(X).max()
        if scale == 0.0:
            return None
        y = X / scale
        return y / np.linalg.norm(y)

    def _update(self, Y: np.ndarray) -> None:
        # DetUpdate (Algorithm 3) with the new instances in the columns of Y, shape (num_features, n).
        D = np.hstack([self.sketch_, Y])  # (num_features, num_columns + n)
        U, s, _ = np.linalg.svd(D, full_matrices=False)
        ell = self.sketch_size_
        shrink = s[ell - 1] ** 2 if s.size >= ell else 0.0
        self.sketch_ = U[:, :ell] * np.sqrt(np.maximum(s[:ell] ** 2 - shrink, 0.0))
        # The singular vectors with nonzero singular values, as np.linalg.matrix_rank decides.
        k = min(self.num_components_, int((s > s[0] * max(D.shape) * np.finfo(s.dtype).eps).sum()))
        self.components_ = U[:, :k]

    def _residual(self, y: np.ndarray) -> float:
        U = self.components_
        return float(np.linalg.norm(y - U @ (U.T @ y)))

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
        x = self._normalize(X)
        if x is None or (self.threshold is not None and self._residual(x) > self.threshold):
            return self
        self._update(x[:, None])

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the instance by the norm of its residual after the projection on the top singular vectors.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: The score of the instance, between 0 and 1. Higher scores represent more anomalous instances.
        """
        if self.to_init:
            self._init_model(X.shape[0])
        x = self._normalize(X)
        return 0.0 if x is None else self._residual(x)
