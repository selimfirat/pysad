"""Score combination functions for the ensemblers.

Copied from combo 0.1.3 (``combo/models/score_comb.py``, https://github.com/yzhao062/combo),
the functions PyOD's ``pyod.models.combination`` wraps, so that pysad does not need combo,
which is only published as a source distribution. The logic is unchanged. For licensing
information, see the end of this file.
"""

import numpy as np
from numpy.random import RandomState
from pyod.utils.utility import check_parameter
from sklearn.utils import check_array, shuffle
from sklearn.utils.random import sample_without_replacement


def average(scores, estimator_weights=None):
    """Combines the scores of multiple estimators by their (weighted) average.

    Args:
        scores (np.float64 array of shape (num_instances, num_estimators)): The scores of the estimators.
        estimator_weights (np.float64 array of shape (1, num_estimators)): The weight of each estimator (Default=None).

    Returns:
        np.float64 array of shape (num_instances,): The combined scores.
    """
    scores = check_array(scores)

    if estimator_weights is not None:
        if estimator_weights.shape != (1, scores.shape[1]):
            raise ValueError(
                f"Bad input shape of estimator_weight: (1, {scores.shape[1]}),"
                f"and {estimator_weights.shape} received"
            )

        # (d1*w1 + d2*w2 + ...+ dn*wn)/(w1+w2+...+wn)
        scores = np.sum(np.multiply(scores, estimator_weights), axis=1) / np.sum(estimator_weights)
        return scores.ravel()

    else:
        return np.mean(scores, axis=1).ravel()


def maximization(scores):
    """Combines the scores of multiple estimators by their maximum.

    Args:
        scores (np.float64 array of shape (num_instances, num_estimators)): The scores of the estimators.

    Returns:
        np.float64 array of shape (num_instances,): The combined scores.
    """
    scores = check_array(scores)
    return np.max(scores, axis=1).ravel()


def median(scores):
    """Combines the scores of multiple estimators by their median.

    Args:
        scores (np.float64 array of shape (num_instances, num_estimators)): The scores of the estimators.

    Returns:
        np.float64 array of shape (num_instances,): The combined scores.
    """
    scores = check_array(scores)
    return np.median(scores, axis=1).ravel()


def aom(scores, n_buckets=5, method="static", bootstrap_estimators=False, random_state=None):
    """Average of maximum :cite:`aggarwal2015theoretical`: splits the estimators into buckets, takes the maximum score in each bucket and averages these maxima.

    Args:
        scores (np.float64 array of shape (num_instances, num_estimators)): The scores of the estimators.
        n_buckets (int): The number of buckets (Default=5).
        method (str): {'static', 'dynamic'}, if 'dynamic', build buckets randomly with dynamic bucket size (Default='static').
        bootstrap_estimators (bool): Whether estimators are drawn with replacement (Default=False).
        random_state (int, np.random.RandomState or None): The seed or random number generator for the bucket assignment. None uses the global NumPy random state (Default=None).

    Returns:
        np.float64 array of shape (num_instances,): The combined scores.
    """
    return _aom_moa_helper("AOM", scores, n_buckets, method, bootstrap_estimators, random_state)


def moa(scores, n_buckets=5, method="static", bootstrap_estimators=False, random_state=None):
    """Maximum of average :cite:`aggarwal2015theoretical`: splits the estimators into buckets, averages the scores in each bucket and takes the maximum of these averages.

    Args:
        scores (np.float64 array of shape (num_instances, num_estimators)): The scores of the estimators.
        n_buckets (int): The number of buckets (Default=5).
        method (str): {'static', 'dynamic'}, if 'dynamic', build buckets randomly with dynamic bucket size (Default='static').
        bootstrap_estimators (bool): Whether estimators are drawn with replacement (Default=False).
        random_state (int, np.random.RandomState or None): The seed or random number generator for the bucket assignment. None uses the global NumPy random state (Default=None).

    Returns:
        np.float64 array of shape (num_instances,): The combined scores.
    """
    return _aom_moa_helper("MOA", scores, n_buckets, method, bootstrap_estimators, random_state)


def _aom_moa_helper(mode, scores, n_buckets, method, bootstrap_estimators, random_state):
    if mode != "AOM" and mode != "MOA":
        raise NotImplementedError(f"{mode} is not implemented")

    scores = check_array(scores)
    n_estimators = scores.shape[1]
    check_parameter(
        n_buckets, 2, n_estimators, include_left=True, include_right=True, param_name="n_buckets"
    )

    scores_buckets = np.zeros([scores.shape[0], n_buckets])

    if method == "static":
        n_estimators_per_bucket = int(n_estimators / n_buckets)
        if n_estimators % n_buckets != 0:
            raise ValueError(
                "n_estimators / n_buckets has a remainder. Not allowed in static mode."
            )

        if not bootstrap_estimators:
            # shuffle the estimator order
            shuffled_list = shuffle(list(range(0, n_estimators, 1)), random_state=random_state)

            head = 0
            for i in range(0, n_estimators, n_estimators_per_bucket):
                tail = i + n_estimators_per_bucket
                batch_ind = int(i / n_estimators_per_bucket)
                if mode == "AOM":
                    scores_buckets[:, batch_ind] = np.max(
                        scores[:, shuffled_list[head:tail]], axis=1
                    )
                else:
                    scores_buckets[:, batch_ind] = np.mean(
                        scores[:, shuffled_list[head:tail]], axis=1
                    )

                # increment index
                head = head + n_estimators_per_bucket
        else:
            for i in range(n_buckets):
                ind = sample_without_replacement(
                    n_estimators, n_estimators_per_bucket, random_state=random_state
                )
                if mode == "AOM":
                    scores_buckets[:, i] = np.max(scores[:, ind], axis=1)
                else:
                    scores_buckets[:, i] = np.mean(scores[:, ind], axis=1)

    elif method == "dynamic":  # random bucket size
        for i in range(n_buckets):
            # the number of estimators in a bucket should be 2 - n/2
            max_estimator_per_bucket = RandomState(seed=random_state).randint(
                2, int(n_estimators / 2)
            )
            ind = sample_without_replacement(
                n_estimators, max_estimator_per_bucket, random_state=random_state
            )
            if mode == "AOM":
                scores_buckets[:, i] = np.max(scores[:, ind], axis=1)
            else:
                scores_buckets[:, i] = np.mean(scores[:, ind], axis=1)

    else:
        raise NotImplementedError(f"{method} is not implemented")

    if mode == "AOM":
        return np.mean(scores_buckets, axis=1)
    else:
        return np.max(scores_buckets, axis=1)


# The functions above are adapted from combo 0.1.3, which is distributed under the
# following license:
#
# BSD 2-Clause License
#
# Copyright (c) 2019, Yue Zhao
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
