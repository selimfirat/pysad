from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax

from pysad.core.base_model import BaseModel


class ZeroShotEmbeddingDetector(BaseModel):
    """Zero-shot OOD detector using semantic anchors (Maximum Concept Matching) :cite:`ming2022delving`.

    The detector takes pre-computed embeddings as input and compares each instance against
    a fixed set of semantic anchors (text prototypes). The anomaly score is based on the
    Maximum Concept Matching (MCM) score from the paper.

    For an input embedding, the cosine similarity to each semantic anchor is computed.
    A softmax with temperature is applied to the similarities, and the negative maximum
    softmax probability is returned as the anomaly score. Higher scores indicate more
    anomalous instances (further from all semantic anchors).

    The semantic anchors are fixed and provided at initialization. They typically
    represent the text embeddings of known in-distribution class names, but can be
    any set of reference points in the embedding space.

    This implementation follows the zero-shot OOD detection approach of Ming et al. (NeurIPS 2022)
    which uses CLIP vision-language features. The detector itself does not include CLIP;
    it operates on pre-computed embeddings that are assumed to be in the same space as
    the semantic anchors.

    Args:
        semantic_anchors (np.ndarray): Array of shape (num_anchors, embedding_dim) containing
            the semantic anchor embeddings. Each row is an anchor (for example, text prototype).
        temperature (float): Temperature parameter for softmax scaling. Lower values
            sharpen the distribution, higher values flatten it. Default=1.0.
        normalize (bool): Whether to L2-normalize input embeddings and anchors before
            computing cosine similarity. If True, assumes anchors are not pre-normalized.
            Default=True.
    """

    __bibtex__ = """
@inproceedings{ming2022delving,
  title={Delving into Out-of-Distribution Detection with Vision-Language Representations},
  author={Ming, Yifei and Cai, Ziyang and Gu, Jiuxiang and Sun, Yiyou and Li, Wei and Li, Yixuan},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  pages={35087--35102},
  year={2022}
}
"""

    def __init__(
        self,
        semantic_anchors: NDArray[np.float64],
        temperature: float = 1.0,
        normalize: bool = True,
    ) -> None:
        self.semantic_anchors = semantic_anchors
        self.temperature = temperature
        self.normalize = normalize
        self._anchors_normalized: NDArray[np.float64] | None = None
        self._embedding_dim: int | None = None

    def _init_model(self, embedding_dim: int) -> None:
        if self.semantic_anchors.shape[1] != embedding_dim:
            raise ValueError(
                f"Semantic anchors dimension ({self.semantic_anchors.shape[1]}) "
                f"does not match input embedding dimension ({embedding_dim})."
            )
        self._embedding_dim = embedding_dim
        if self.normalize:
            norms = np.linalg.norm(self.semantic_anchors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._anchors_normalized = self.semantic_anchors / norms
        else:
            self._anchors_normalized = self.semantic_anchors.copy()

    def _normalize(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.normalize:
            return X
        norm = np.linalg.norm(X)
        if norm == 0:
            return X
        return np.asarray(X / norm, dtype=np.float64)

    def fit_partial(
        self, X: NDArray[np.float64], y: int | None = None
    ) -> ZeroShotEmbeddingDetector:
        """This method does not update the model. The semantic anchors are fixed.

        Args:
            X (np.ndarray): The embedding to fit. Shape (embedding_dim,).
            y (int | None): Ignored.

        Returns:
            ZeroShotEmbeddingDetector: The model itself.
        """
        if self._anchors_normalized is None:
            self._init_model(X.shape[0])
        return self

    def score_partial(self, X: NDArray[np.float64]) -> float:
        """Scores the anomalousness of the next embedding using MCM.

        The score is the negative maximum softmax probability over cosine similarities
        to the semantic anchors. Higher scores mean more anomalous (further from all anchors).

        Args:
            X (np.ndarray): The embedding to score. Shape (embedding_dim,).

        Returns:
            float: Anomaly score (negative max softmax probability).
        """
        if self._anchors_normalized is None:
            self._init_model(X.shape[0])

        anchors = self._anchors_normalized
        assert anchors is not None
        x_norm = self._normalize(X)
        similarities = anchors @ x_norm
        probs = softmax(similarities / self.temperature)
        max_prob = np.max(probs)
        return float(-max_prob)

    def fit_score_partial(self, X: NDArray[np.float64], y: int | None = None) -> float:
        """Scores the embedding, then fits (no-op for this detector).

        Args:
            X (np.ndarray): The embedding to score and fit. Shape (embedding_dim,).
            y (int | None): Ignored.

        Returns:
            float: Anomaly score.
        """
        if self._anchors_normalized is None:
            self._init_model(X.shape[0])
        return self.score_partial(X)
