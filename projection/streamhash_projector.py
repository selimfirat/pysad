#!/usr/bin/env python

import math
import numpy as np
import random
import mmh3


class StreamhashProjector:

    def __init__(self, n_components, density=1 / 3.0, random_state=None):
        self.keys = np.arange(0, n_components, 1)
        self.constant = np.sqrt(1. / density) / np.sqrt(n_components)
        self.density = density
        self.n_components = n_components
        random.seed(random_state)

    def fit_transform(self, X, feature_names=None):
        nsamples = X.shape[0]
        ndim = X.shape[1]
        if feature_names is None:
            feature_names = [str(i) for i in range(ndim)]

        R = np.array([[self._hash_string(k, f)
                       for f in feature_names]
                      for k in self.keys])

        # check if density matches
        # print "R", np.sum(R.ravel() == 0)/float(len(R.ravel()))

        Y = np.dot(X, R.T)
        return Y

    def transform(self, X, feature_names=None):
        return self.fit_transform(X, feature_names)

    def _hash_string(self, k, s):
        hash_value = int(mmh3.hash(s, signed=False, seed=k)) / (2.0 ** 32 - 1)
        s = self.density
        if hash_value <= s / 2.0:
            return -1 * self.constant
        elif hash_value <= s:
            return self.constant
        else:
            return 0
