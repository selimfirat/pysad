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
from projection.base_projector import BaseProjector
from projection.grouse_projector import Tracker


class Petrels(Tracker):
    """
    This class is an implementation of the "PETRELS: Parallel Subspace
    Estimation and Tracking by Recursive Least Squares from Partial
    Observations" presented by Chi et al. in
    http://arxiv.org/abs/1207.6353v2.

    This class assumes the use of Numpy vectors.
    """

    def __init__(self, ambient_dim, rank, forgetting=0.98):
        """
        Keyword arguments:
        ambient_dim -- ambient dimension of the observations
        rank -- estimate of the rank
        forgetting -- forgetting (discount) factor
        """
        self.ambient_dim = ambient_dim
        self.rank = rank
        self.forgetting = forgetting

        self._delta = 1E5

        # Initialize random subspace
        self.U = np.random.rand(self.ambient_dim, self.rank)

        # Initialize recursive least squares estimates
        self.R = []
        for i in range(self.ambient_dim):
            self.R.append(np.identity(self.rank) * self._delta)

    def consume(self, ob_vec, sample_vec):
        forgetting_inv = 1.0 / self.forgetting

        # Find the projection
        proj = self._project(ob_vec, sample_vec)

        # Update subspace rows for the dimensions we have observations in
        observed_dims = np.nonzero(sample_vec)[0]
        for idx in observed_dims:
            ob = ob_vec[idx]

            # Calculate beta
            beta = 1 + proj.T @ self.R[idx] @ proj
            beta_inv = 1.0 / beta

            # Calculate v
            v = self.R[idx] @ proj

            # Update R
            self.R[idx] *= forgetting_inv
            self.R[idx] -= beta_inv * v * v.T

            update = ((ob - (proj.T @ self.U.T[:, idx])) * self.R[idx] @ proj).reshape(self.rank)

            self.U.T[:, idx] += update


class PetrelsProjector(BaseProjector):

    def __init__(self, n_components, forgetting=0.98, **kwargs):

        super().__init__(n_components, **kwargs)

        self.tracker = None
        self.forgetting = forgetting

        self.ss = None

    def fit_partial(self, X):

        x = X.reshape(-1, 1)

        if self.tracker is None:
            self.tracker = Petrels(X.shape[1], self.n_components, self.forgetting)
            self.ss = np.ones_like(x)

        self.tracker.consume(x, self.ss)

        return self

    def transform_partial(self, X):

        x = X.reshape(-1, 1)

        proj = self.tracker._project(x, self.ss).T

        return proj
