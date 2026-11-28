from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import ClusterNode, linkage, to_tree
from scipy.special import expit
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class _CorClust:
    """Incremental correlation-based feature clusterer for KitNET."""

    def __init__(self, num_features: int) -> None:
        self.n = num_features
        self.c: NDArray[np.float64] = np.zeros(num_features, dtype=np.float64)
        self.c_r: NDArray[np.float64] = np.zeros(num_features, dtype=np.float64)
        self.c_rs: NDArray[np.float64] = np.zeros(num_features, dtype=np.float64)
        self.C: NDArray[np.float64] = np.zeros((num_features, num_features), dtype=np.float64)
        self.n_updates = 0

    def update(self, x: NDArray[np.float64]) -> None:
        self.n_updates += 1
        self.c += x
        c_rt = x - self.c / self.n_updates
        self.c_r += c_rt
        self.c_rs += c_rt**2
        self.C += np.outer(c_rt, c_rt)

    def corr_dist(self) -> NDArray[np.float64]:
        c_rs_sqrt = np.sqrt(self.c_rs)
        denom = np.outer(c_rs_sqrt, c_rs_sqrt)
        denom[denom == 0.0] = 1e-100
        dist: NDArray[np.float64] = 1.0 - self.C / denom
        dist[dist < 0.0] = 0.0
        return dist

    def cluster(self, max_clust: int) -> list[list[int]]:
        if self.n == 1:
            return [[0]]
        dist = self.corr_dist()
        z_mat = linkage(dist[np.triu_indices(self.n, 1)])
        bound = min(max(max_clust, 1), self.n)
        root = to_tree(z_mat)
        if not isinstance(root, ClusterNode):
            raise RuntimeError("Expected a ClusterNode root from hierarchical linkage.")
        return self._break_clusters(root, bound)

    @staticmethod
    def _break_clusters(dendro: ClusterNode, max_clust: int) -> list[list[int]]:
        # Iterative stack avoids Python recursion-limit failures on chained single-linkage trees.
        clusters: list[list[int]] = []
        stack: list[ClusterNode] = [dendro]
        while stack:
            node = stack.pop()
            if node.count <= max_clust:
                clusters.append([int(idx) for idx in node.pre_order()])
            else:
                right = node.get_right()
                left = node.get_left()
                if right is not None:
                    stack.append(right)
                if left is not None:
                    stack.append(left)
        return clusters


class _DenoisingAutoencoder:
    """Single online tied-weight autoencoder used in KitNET's ensemble and output layers."""

    def __init__(
        self,
        n_visible: int,
        lr: float,
        hidden_ratio: float,
        rng: np.random.RandomState,
    ) -> None:
        n_hidden = int(math.ceil(n_visible * hidden_ratio))
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        self.n_trained = 0

        self.norm_max: NDArray[np.float64] = np.full(n_visible, -np.inf, dtype=np.float64)
        self.norm_min: NDArray[np.float64] = np.full(n_visible, np.inf, dtype=np.float64)

        bound = 1.0 / n_visible
        self.W: NDArray[np.float64] = np.asarray(
            rng.uniform(low=-bound, high=bound, size=(n_visible, n_hidden)),
            dtype=np.float64,
        )
        self.hbias: NDArray[np.float64] = np.zeros(n_hidden, dtype=np.float64)
        self.vbias: NDArray[np.float64] = np.zeros(n_visible, dtype=np.float64)

    def normalize(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # Constant training features are shifted without dividing by the 1e-16 guard.
        scale = self.norm_max - self.norm_min + 1e-16
        scale[self.norm_max == self.norm_min] = 1.0
        return (x - self.norm_min) / scale

    def train(self, x: NDArray[np.float64]) -> float:
        self.n_trained += 1
        np.maximum(self.norm_max, x, out=self.norm_max)
        np.minimum(self.norm_min, x, out=self.norm_min)

        x_norm = self.normalize(x)
        y = np.asarray(expit(x_norm @ self.W + self.hbias), dtype=np.float64)
        z = np.asarray(expit(y @ self.W.T + self.vbias), dtype=np.float64)

        err_v = x_norm - z
        err_h = (err_v @ self.W) * y * (1.0 - y)

        self.W += self.lr * (np.outer(x_norm, err_h) + np.outer(err_v, y))
        # Upstream KitNET-py uses np.mean(..., axis=0) on 1-D error vectors (scalar mean).
        self.hbias += self.lr * float(err_h.mean())
        self.vbias += self.lr * float(err_v.mean())
        return float(np.sqrt(np.mean(err_v**2)))

    def execute(self, x: NDArray[np.float64]) -> float:
        x_norm = self.normalize(x)
        y = np.asarray(expit(x_norm @ self.W + self.hbias), dtype=np.float64)
        z = np.asarray(expit(y @ self.W.T + self.vbias), dtype=np.float64)
        return float(np.sqrt(((x_norm - z) ** 2).mean()))


class _KitNET:
    """Two-layer online autoencoder ensemble from KitNET-py.

    This implementation matches KitNET-py's behavior: no autoencoder training occurs
    during the feature mapping grace period. Only the correlation matrix is updated.
    Autoencoders are created and trained only after the feature mapping grace period ends.
    """

    def __init__(
        self,
        num_features: int,
        max_autoencoder_size: int = 10,
        fm_grace_period: int | None = None,
        ad_grace_period: int = 10000,
        learning_rate: float = 0.1,
        hidden_ratio: float = 0.75,
        feature_map: list[list[int]] | None = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.ad_grace_period = ad_grace_period
        self.fm_grace_period = ad_grace_period if fm_grace_period is None else fm_grace_period
        self.m = 1 if max_autoencoder_size <= 0 else max_autoencoder_size
        self.lr = learning_rate
        self.hr = hidden_ratio
        self.n = num_features
        self.random_state = random_state

        self.n_trained = 0
        self.n_executed = 0
        self.feature_map = feature_map
        self.fm = _CorClust(self.n)
        self.active_map: list[list[int]] | None = None
        self.ensemble_layer: list[_DenoisingAutoencoder] = []
        self.output_layer: _DenoisingAutoencoder | None = None

    def in_grace(self) -> bool:
        return self.n_trained < self.fm_grace_period + self.ad_grace_period

    def train(self, x: NDArray[np.float64]) -> None:
        if self.n_trained < self.fm_grace_period and self.feature_map is None:
            self.fm.update(x)
        elif self.output_layer is None:
            if self.feature_map is None:
                self.feature_map = self.fm.cluster(self.m)
            self._create_autoencoders(self.feature_map)

        if self.active_map is not None and self.output_layer is not None:
            scores_l1 = np.empty(len(self.ensemble_layer), dtype=np.float64)
            for idx, ae in enumerate(self.ensemble_layer):
                scores_l1[idx] = ae.train(x[self.active_map[idx]])
            self.output_layer.train(scores_l1)
        self.n_trained += 1

    def execute(self, x: NDArray[np.float64]) -> float:
        if self.output_layer is None or self.active_map is None:
            raise RuntimeError("KitNET autoencoders have not trained on an instance yet.")
        self.n_executed += 1
        scores_l1 = np.empty(len(self.ensemble_layer), dtype=np.float64)
        for idx, ae in enumerate(self.ensemble_layer):
            scores_l1[idx] = ae.execute(x[self.active_map[idx]])
        return self.output_layer.execute(scores_l1)

    def _create_autoencoders(self, feature_map: list[list[int]]) -> None:
        rng = check_random_state(self.random_state)
        self.active_map = feature_map
        self.ensemble_layer = [
            _DenoisingAutoencoder(len(cluster), self.lr, self.hr, rng) for cluster in feature_map
        ]
        self.output_layer = _DenoisingAutoencoder(len(feature_map), self.lr, self.hr, rng)


class KitNet(BaseModel):
    """Scores each instance by the reconstruction error of an online ensemble of small autoencoders (KitNET) :cite:`mirsky2018kitsune`.

    This model uses the implementation from `KitNET-py <https://github.com/ymirsky/KitNET-py>`_ with one change: the 0-1 normalization only shifts an input that was constant in training, which KitNET-py divides by 1e-16.

    As in the paper, the model learns only during two grace periods at the start of the stream and then only scores. The first ``grace_feature_mapping`` instances learn which features go to which autoencoder; during this period the model returns ``0.0`` and does not train autoencoders. The next ``grace_anomaly_detector`` instances train the autoencoders, and their scores come from autoencoders that are still training. Choose both so that their sum is a small part of the stream and still covers its normal behavior. The defaults, 100 and 1,000 instances, keep the 1:10 ratio of the KitNET-py example (5,000 and 50,000 on a capture of 100,000 packets), scaled down so that a stream of a few hundred instances gets real scores. For a long stream whose start is free of anomalies, such as the paper's network captures (it trains on the first million packets), longer grace periods give the autoencoders more normal behavior to learn.

    On datasets smaller than the combined grace periods, the model will not finish training and scores will be unreliable (returning ``0.0`` during the feature mapping period). For the ODDS benchmark datasets, grace periods should be set proportionally to the dataset size (e.g., 5% and 50% of the stream length).

    :meth:`explain_partial` returns the per-feature attribution weights of one instance.
    Each ensemble autoencoder reconstructs the features it maps, and its squared reconstruction
    error is attributed to those features. The weights are these per-feature errors, normalized
    to sum to one, so a larger weight means the autoencoders that use that feature reconstruct it
    poorly. The weights have shape ``(num_features,)``. Before the first autoencoders exist, the
    method returns zeros of shape ``(num_features,)``. This is the case on the first instance,
    before it has been fitted.
    This attribution scheme is an addition of pysad. It is not part of the cited paper.
    Both :meth:`explain_partial` and :meth:`score_partial` raise a ``ValueError`` for an
    instance with NaN.

    Args:
        max_size_ae (int): The maximum size of any autoencoder in the ensemble layer (Default=10).
        grace_feature_mapping (int): The number of instances used to learn the feature mapping. If ``None``, the feature mapping and anomaly detector grace periods are equal (Default=100).
        grace_anomaly_detector (int): The number of instances used to train the autoencoders after the feature mapping is learned (Default=1000).
        learning_rate (float): The default stochastic gradient descent learning rate for all autoencoders in the KitNET instance (Default=0.1).
        hidden_ratio (float): The default ratio of hidden to visible neurons. E.g., 0.75 will cause roughly a 25% compression in the hidden layer (Default=0.75).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the initial weights of the autoencoders, which all share it. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self,
        max_size_ae: int = 10,
        grace_feature_mapping: int | None = 100,
        grace_anomaly_detector: int = 1000,
        learning_rate: float = 0.1,
        hidden_ratio: float = 0.75,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.grace_feature_mapping = grace_feature_mapping
        self.hidden_ratio = hidden_ratio
        self.learning_rate = learning_rate
        self.max_size_ae = max_size_ae
        self.grace_anomaly_detector = grace_anomaly_detector
        self.random_state = random_state
        self.model: _KitNET | None = None
        self.feature_map_: list[list[int]] | None = None

    def fit_partial(self, X: NDArray[np.float64], y: int | None = None) -> KitNet:
        """Fits the model to the next instance.

        The first ``grace_feature_mapping`` instances learn the feature mapping;
        the next ``grace_anomaly_detector`` instances train the autoencoders.
        Before the first instance, :meth:`score_partial` returns ``0.0``.
        During the feature mapping grace period, :meth:`score_partial` returns ``0.0``
        and no autoencoder training occurs. During the anomaly detector grace period,
        scores come from autoencoders that are still training, so they are unreliable
        for alerting. After both grace periods the model stops learning and only scores.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        if self.model is None:
            self.num_features = X.shape[0]
            self.model = _KitNET(
                self.num_features,
                self.max_size_ae,
                self.grace_feature_mapping,
                self.grace_anomaly_detector,
                self.learning_rate,
                self.hidden_ratio,
                random_state=self.random_state,
            )
        # After both grace periods KitNET only scores, which is score_partial's job.
        if self.model.in_grace():
            self.model.train(X)
            self.feature_map_ = self.model.feature_map

        return self

    def score_partial(self, X: NDArray[np.float64]) -> float:
        """Scores the anomalousness of the next instance.

        Returns ``0.0`` before the first instance and during the feature mapping grace
        period. During the anomaly detector grace period, scores come from partially
        trained autoencoders and are unreliable for alerting.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if self.model is None or self.model.output_layer is None:
            return 0.0
        return self.model.execute(X)

    def explain_partial(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Returns per-feature attribution weights for the anomaly score of the next instance.

        Each ensemble autoencoder reconstructs the features it maps, and its squared reconstruction
        error is attributed to those features. The weights are these per-feature errors, normalized
        to sum to one, so a larger weight means the autoencoders that use that feature reconstruct
        it poorly.

        Like ``score_partial``, it does not update the model or record the instance. Before the
        first autoencoders exist, the method returns zeros of shape ``(num_features,)``. This is
        the case on the first instance, before it has been fitted.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to explain.

        Returns:
            np.float64 array of shape (num_features,): Non-negative attribution weights, one per
            feature, summing to one (zeros before the first autoencoders exist).

        Raises:
            ValueError: If `X` holds NaN.
        """
        x = np.asarray(X)
        if x.dtype.kind in "fc" and bool(np.isnan(x).any()):
            raise ValueError(f"{type(self).__name__} does not accept NaN values.")

        model = self.model
        if model is None or model.active_map is None or not model.ensemble_layer:
            return np.zeros(x.shape[0], dtype=np.float64)

        attribution: NDArray[np.float64] = np.zeros(x.shape[0], dtype=np.float64)
        for autoencoder, features in zip(model.ensemble_layer, model.active_map, strict=True):
            subset = np.asarray(list(features), dtype=np.intp)
            normalized = autoencoder.normalize(x[subset])
            hidden = np.asarray(
                expit(normalized @ autoencoder.W + autoencoder.hbias), dtype=np.float64
            )
            reconstructed = np.asarray(
                expit(hidden @ autoencoder.W.T + autoencoder.vbias), dtype=np.float64
            )
            np.add.at(attribution, subset, (normalized - reconstructed) ** 2)

        total = float(attribution.sum())
        if total > 0.0:
            attribution = attribution / total

        return attribution

    def score_explain_partial(self, X: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Returns the anomaly score and per-feature attribution for the next instance.

        The explanation attributes the reconstruction error of each ensemble autoencoder
        to the features it maps, normalized to sum to one. Higher attribution weights
        mean the autoencoders using that feature reconstruct it poorly, indicating
        that feature contributes more to the anomaly score.

        Like ``score_partial``, it does not update the model or record the instance.
        Before the first autoencoders exist, the method returns 0.0 and zeros of shape
        ``(num_features,)``.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score and explain.

        Returns:
            tuple: (score, attribution) where
                score (float): The anomalousness score of the input instance.
                attribution (np.float64 array of shape (num_features,)): Non-negative
                attribution weights, one per feature, summing to one (zeros before the
                first autoencoders exist).

        Raises:
            ValueError: If `X` holds NaN.
        """
        x = np.asarray(X)
        if x.dtype.kind in "fc" and bool(np.isnan(x).any()):
            raise ValueError(f"{type(self).__name__} does not accept NaN values.")

        score = self.score_partial(X)
        attribution = self.explain_partial(X)
        return score, attribution


# _CorClust, _DenoisingAutoencoder, and _KitNET are adapted from KitNET-py
# (https://github.com/ymirsky/KitNET-py) and Yusuke Sugomori's DeepLearning
# (https://github.com/yusugomori/DeepLearning), distributed under the MIT License:
#
# Copyright (c) 2017 Yisroel Mirsky
# Copyright (c) 2017 Yusuke Sugomori
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
