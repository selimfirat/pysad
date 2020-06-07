#!/usr/bin/env python

import math
import numpy as np
import random
import mmh3

from projection.base_projector import BaseProjector


class StreamhashProjector(BaseProjector):

    """

    Reference: https://github.com/cmuxstream/cmuxstream-core
    """
    def __init__(self, n_components, density=1 / 3.0, **kwargs):
        super().__init__(n_components, **kwargs)
        self.keys = np.arange(0, n_components, 1)
        self.constant = np.sqrt(1. / density) / np.sqrt(n_components)
        self.density = density
        self.n_components = n_components

    def fit_partial(self, X):

        return self

    def transform_partial(self, X):
        X = X.reshape(1, -1)

        ndim = X.shape[1]

        feature_names = [str(i) for i in range(ndim)]

        R = np.array([[self._hash_string(k, f)
                       for f in feature_names]
                      for k in self.keys])

        Y = np.dot(X, R.T).squeeze()

        return Y

    def _hash_string(self, k, s):
        hash_value = int(mmh3.hash(s, signed=False, seed=k)) / (2.0 ** 32 - 1)
        s = self.density
        if hash_value <= s / 2.0:
            return -1 * self.constant
        elif hash_value <= s:
            return self.constant
        else:
            return 0
