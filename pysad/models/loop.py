from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import erf

from pysad.core.base_model import BaseModel


class LocalOutlierProbability(BaseModel):
    """Scores each instance by its local outlier probability against an initial reference set :cite:`kriegel2009loop`.

    The model uses the incremental and approximate version of LoOP by Hamlet et al. :cite:`hamlet2017incremental`, as the PyNomaly library :cite:`constantinou2018pynomaly` implements it. The model fits LoOP once on ``initial_X``. Then it scores each instance with values from that fit, for example a global expected value of the probabilistic distance. The model does not learn from the stream, so ``fit_partial`` does nothing.

    Args:
        initial_X (np.float64 array of shape (num_instances, num_features)): Initial training data to calibrate the model. It must have more than ``num_neighbors`` instances.
        num_neighbors (int): The total number of neighbors to consider w.r.t. each sample (Default=10).
        extent (int): The statistical extent lambda, the number of standard deviations from the mean that the probabilistic distance covers, usually 1, 2 or 3 (Default=3).
    """

    def __init__(self, initial_X: np.ndarray, num_neighbors: int = 10, extent: int = 3) -> None:
        # Adapted from LocalOutlierProbability.fit of PyNomaly 0.4.0 (PyNomaly/loop.py,
        # https://github.com/vc1492a/PyNomaly) for a single cluster. For licensing information,
        # see the end of this file.
        self.initial_X = np.asarray(initial_X, dtype=np.float64)
        self.num_neighbors = num_neighbors
        self.extent = extent

        distances = cdist(self.initial_X, self.initial_X)
        np.fill_diagonal(distances, np.inf)
        neighbors = np.argpartition(distances, num_neighbors, axis=1)[:, :num_neighbors]
        neighbor_distances = np.take_along_axis(distances, neighbors, axis=1)
        prob_distances = extent * np.sqrt((neighbor_distances**2).sum(axis=1) / num_neighbors)
        prob_distances_ev = prob_distances[neighbors].mean(axis=1)
        plofs = np.asarray(self._prob_outlier_factor(prob_distances, prob_distances_ev))
        self.prob_distance_ev_ = prob_distances_ev.mean()
        self.norm_prob_outlier_factor_ = extent * np.sqrt((plofs**2).sum() / plofs.size)

    @staticmethod
    def _prob_outlier_factor(
        prob_distance: np.ndarray | float, prob_distance_ev: np.ndarray | float
    ) -> np.ndarray | float:
        if np.all(prob_distance == prob_distance_ev):
            return np.zeros(np.shape(prob_distance))
        prob_distance_ev = np.where(prob_distance_ev == 0.0, 1.0e-8, prob_distance_ev)
        return prob_distance / prob_distance_ev - 1.0

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> LocalOutlierProbability:
        """Updates the model with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        # Adapted from LocalOutlierProbability.stream of PyNomaly 0.4.0, with the same operations so
        # that the scores are bit-identical: one dot product per instance on its own difference
        # array (BLAS rounds row views differently), and the neighbor distances in the order of
        # the replace-the-largest loop, which their sum depends on.
        neighbor_distances = np.full(self.num_neighbors, 9e10)
        for instance in self.initial_X:
            diff = instance - X
            distance = np.dot(diff, diff) ** 0.5
            largest = neighbor_distances.argmax()
            if distance < neighbor_distances[largest]:
                neighbor_distances[largest] = distance
        ssd = (neighbor_distances**2).sum()
        prob_distance = self.extent * np.sqrt(ssd / self.num_neighbors)
        plof = self._prob_outlier_factor(prob_distance, self.prob_distance_ev_)
        if plof == self.norm_prob_outlier_factor_:
            return 0.0
        # When the reference PLOFs have no spread (e.g. identical initial instances), the
        # normalization is zero and the score is its limit, 1 for any positive PLOF.
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(
                np.maximum(0.0, erf(plof / (self.norm_prob_outlier_factor_ * np.sqrt(2.0))))
            )


# The fit and score code above is adapted from PyNomaly 0.4.0 and rewritten in vectorized
# NumPy. PyNomaly is distributed under the following license:
#
# Copyright 2017 Valentino Constantinou.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
