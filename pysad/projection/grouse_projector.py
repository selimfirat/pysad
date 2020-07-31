# Copyright 2015 Gregory Hasseler
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import random
import abc
from tqdm import tqdm

from projection.base_projector import BaseProjector


class Tracker(object, metaclass=abc.ABCMeta):
    """
    Abstract base class for tracker implementations.
    """

    @abc.abstractclassmethod
    def consume(self, ob_vec, sample_vec):
        """Consume an observation and corresponding sampling vector.

        Keyword arguments:
        ob_vec -- the observation vector with unobserved attributes not present (NumPy vector)
        sample_vec -- the sampling vector (NumPy vector)
        """

        raise NotImplementedError("users must define consume to use this base class")

    def _project(self, ob_vec, sample_vec):
        """Find the projection of the observation vector (ob_vec) into the subspace.
        Keyword arguments:
        ob_vec: observation vector with possibly missing data. Noise entries should be marked in sample_vec
        sample_vec: vector indicating missing data entries in ob_vec. Entries corresponding with ob_vec should be 1 where data was recorded and 0 where data was missing
        """

        # Take only the rows of U that correspond to attributes we have observed
        row_indices = np.nonzero(sample_vec)[0]
        U_samp = self.U[row_indices, :]
        ob_vec_samp = ob_vec[row_indices, :]

        # Compute the weights
        projection = np.linalg.lstsq(U_samp, ob_vec_samp, rcond=-1)[0]

        return projection


class Grouse(Tracker):
    """
    This class is an implementation of the Grassmannian Rank-One Update
    Subspace Estimation (GROUSE) subspace tracking algorithm, presented by
    Balzano et al. in  http://arxiv.org/abs/1006.4046

    This class assumes the use of Numpy vectors.
    """

    def __init__(self, ambient_dim, rank, step):
        """
        Keyword arguments:
        ambient_dim -- ambient dimension of the observations
        rank -- estimate of the rank
        step -- step size
        """
        self.ambient_dim = ambient_dim
        self.rank = rank
        self.step = step

        # Initialize our subspace estimate to a random orthogonal matrix
        self.U = np.linalg.qr(np.random.rand(ambient_dim, rank))[0]

    def consume(self, ob_vec, sample_vec, step=None):
        if step is None:
            step = self.step

        # Find our projection into the subspace
        projection = self._project(ob_vec, sample_vec)

        # Find the imputed measurement residual
        row_indices = np.nonzero(sample_vec)[0]
        imputed_measurement = self.U @ projection
        residual = np.zeros((self.ambient_dim, 1))
        residual[row_indices, :] = ob_vec[row_indices, :] - imputed_measurement[row_indices, :]

        sigma = np.linalg.norm(residual) * np.linalg.norm(imputed_measurement)

        normalized_imputed_measurement = np.nan_to_num(imputed_measurement / np.linalg.norm(imputed_measurement))
        normalized_residual = np.nan_to_num(residual / np.linalg.norm(residual))
        normalized_projection = np.nan_to_num(projection.T / np.linalg.norm(projection))

        lhs_inner_update = (np.cos(sigma * step) - 1) * normalized_imputed_measurement
        rhs_inner_update = np.sin(sigma * step) * normalized_residual
        rhs = (lhs_inner_update + rhs_inner_update) @ normalized_projection

        self.U = self.U + rhs


class GrouseProjector(BaseProjector):

    def __init__(self, n_components, step_size=1, **kwargs):

        super().__init__(n_components, **kwargs)
        self.n_components = n_components
        self.tracker = None

        self.step_size = step_size
        self.ss = None

    def fit_partial(self, X):

        x = X.reshape(-1, 1)

        if self.tracker is None:
            self.tracker = Grouse(X.shape[0], self.n_components, step=1)
            self.ss = np.ones_like(x)

        self.tracker.consume(x, self.ss)


        return self

    def transform_partial(self, X):
        x = X.reshape(-1, 1)

        proj = self.tracker._project(x, self.ss).T

        return proj

