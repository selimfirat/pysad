from __future__ import annotations

from typing import ClassVar

import numpy as np

from pysad.core.base_model import BaseModel


class TEDA(BaseModel):
    r"""Typicality and eccentricity data analytics (TEDA) :cite:`angelov2014anomaly`.

    TEDA keeps the number of points :math:`k`, the recursive mean :math:`\mu_k` and the recursive variance :math:`\sigma_k^2` of the stream, with the Euclidean distance:

    .. math::

        \mu_k = \frac{k - 1}{k} \mu_{k-1} + \frac{x_k}{k}, \qquad
        \sigma_k^2 = \frac{k - 1}{k} \sigma_{k-1}^2 + \frac{\lVert x_k - \mu_k \rVert^2}{k - 1}.

    The eccentricity of a point :math:`x` is :math:`\xi_k(x) = 1/k + \lVert \mu_k - x \rVert^2 / (k \sigma_k^2)`. The paper flags :math:`x` when the normalized eccentricity :math:`\xi_k(x) / 2` is above :math:`(m^2 + 1) / (2k)`. The model keeps no window and no sample, so its memory is fixed and one call costs O(num_features). It takes no ``random_state``, because it draws no random numbers.

    As in the paper, the score of :math:`x` uses the statistics that include :math:`x`. ``score_partial`` computes them without changing the model.

    Deviations from the paper:

    - The model returns :math:`\lVert x - \mu_k \rVert^2 / \sigma_k^2` instead of the normalized eccentricity. The paper threshold changes with :math:`k`, so the normalized eccentricities of two points are not comparable. The returned score is above :math:`m^2` exactly when the paper flags the point, so the paper's rule with :math:`m = 3` is the fixed threshold 9.
    - The paper divides by zero when :math:`\sigma_k^2 = 0`, for example at the first point or on a constant stream. The model scores such a point 0.
    """

    __bibtex__: ClassVar[str] = """
@inproceedings{angelov2014anomaly,
  title={Anomaly Detection Based on Eccentricity Analysis},
  author={Angelov, Plamen},
  booktitle={2014 IEEE Symposium on Evolving and Autonomous Learning Systems (EALS)},
  pages={1--8},
  year={2014},
  doi={10.1109/EALS.2014.7009497}
}
"""
    _detector_doc: ClassVar[tuple[str, str, str]] = (
        "Eccentricity from the recursive mean and variance :cite:`angelov2014anomaly`",
        "Nothing",
        "Fixed: the count, the mean and the variance",
    )

    def __init__(self) -> None:
        self.count_ = 0
        self.mean_: np.ndarray | float = 0.0
        self.variance_ = 0.0

    def _updated(self, X: np.ndarray) -> tuple[int, np.ndarray, float, float]:
        """The count, mean, variance and squared distance to the mean after the model fits ``X``."""
        k = self.count_ + 1
        mean = self.mean_ + (X - self.mean_) / k
        sq_dist = float(((X - mean) ** 2).sum())
        variance = (k - 1) / k * self.variance_ + sq_dist / (k - 1) if k > 1 else 0.0

        return k, mean, variance, sq_dist

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> TEDA:
        """Updates the count, the mean and the variance with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        self.count_, self.mean_, self.variance_, _ = self._updated(X)

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the instance with the statistics that include it, without changing the model.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: The squared distance of the instance to the mean, divided by the variance. A score above :math:`m^2` means that the paper flags the instance with the threshold :math:`m`.
        """
        _, _, variance, sq_dist = self._updated(X)

        return sq_dist / variance if variance > 0.0 else 0.0
