from pysad.core.base_model import BaseModel
import numpy as np


class LODA(BaseModel):
    """

        Reference: https://pyod.readthedocs.io/en/latest/_modules/pyod/models/loda.html#LODA
    """

    def __init__(self, num_features, num_bins=10, num_random_cuts=100):

        self.num_features = num_features
        self.n_bins = num_bins
        self.n_random_cuts = num_random_cuts
        self.weights = np.ones(num_random_cuts, dtype=np.float) / num_random_cuts
        self.projections_ = np.random.randn(self.n_random_cuts, num_features)
        self.histograms_ = np.zeros((self.n_random_cuts, self.n_bins))
        self.limits_ = np.zeros((self.n_random_cuts, self.n_bins + 1))

        n_nonzero_components = np.sqrt(self.num_features)
        self.n_zero_components = self.num_features - np.int(n_nonzero_components)

    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X: np.float array of shape (num_features,)
                The instance to fit.
            y: int (Default=None)
                Ignored since the model is unsupervised.

        Returns:
            self: object
                Returns the self.
        """
        X = X.reshape(1, -1)

        for i in range(self.n_random_cuts):
            rands = np.random.permutation(self.num_features)[:self.n_zero_components]
            self.projections_[i, rands] = 0.
            projected_data = self.projections_[i, :].dot(X.T)
            self.histograms_[i, :], self.limits_[i, :] = np.histogram(
                projected_data, bins=self.n_bins, density=False)
            self.histograms_[i, :] += 1e-12
            self.histograms_[i, :] /= np.sum(self.histograms_[i, :])

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X: np.float array of shape (num_features,)
                The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            score: float
                The anomalousness score of the input instance.
        """
        X = X.reshape(1, -1)

        pred_scores = np.zeros([X.shape[0], 1])
        for i in range(self.n_random_cuts):
            projected_data = self.projections_[i, :].dot(X.T)
            inds = np.searchsorted(self.limits_[i, :self.n_bins - 1],
                                   projected_data, side='left')
            pred_scores[:, 0] += -self.weights[i] * np.log(
                self.histograms_[i, inds])
        pred_scores /= self.n_random_cuts

        return pred_scores.ravel()
