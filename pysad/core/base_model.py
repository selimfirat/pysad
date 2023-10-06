from abc import ABC, abstractmethod
from pysad.utils import _iterate
import numpy as np


class BaseModel(ABC):
    """Abstract base class for the models.
    """

    @abstractmethod
    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): The label of the instance (Optional for unsupervised models, default=None).

        Returns:
            object: Returns the self.
        """
        pass

    @abstractmethod
    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        pass

    def fit_score_partial(self, X, y=None):
        """Applies fit_partial and score_partial to the next instance, respectively.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int): The label of the instance (Optional for unsupervised models, default=None).

        Returns:
            float: The anomalousness score of the input instance.
        """
        return self.fit_partial(X, y).score_partial(X)

    def fit(self, X, y=None):
        """Fits the model to all instances in order.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances in order to fit.
            y (int): The labels of the instances in order to fit (Optional for unsupervised models, default=None).

        Returns:
            object: Fitted model.
        """
        for xi, yi in _iterate(X, y):
            self.fit_partial(xi, yi)

        return self

    def score(self, X):
        """Scores all instaces via score_partial iteratively.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances in order to score.

        Returns:
            np.float64 array of shape (num_instances,): The anomalousness scores of the instances in order.
        """
        y_pred = np.empty(X.shape[0], dtype=np.float64)
        for i, (xi, _) in enumerate(_iterate(X)):
            y_pred[i] = self.score_partial(xi)

        return y_pred

    def fit_score(self, X, y=None):
        """This helper method applies fit_score_partial to all instances in order.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances in order to fit.
            y (np.int32 array of shape (num_instances, )): The labels of the instances in order to fit (Optional for unsupervised models, default=None).

        Returns:
            np.float64 array of shape (num_instances,): The anomalousness scores of the instances in order.
        """
        y_pred = np.zeros(X.shape[0], dtype=np.float64)
        for i, (xi, yi) in enumerate(_iterate(X, y)):
            y_pred[i] = self.fit_score_partial(xi, yi)

        return y_pred
