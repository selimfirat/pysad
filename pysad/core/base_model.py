from abc import ABC, abstractmethod
from pysad.utils import _iterate


class BaseModel(ABC):
    """Abstract base class for the models.
    """

    @abstractmethod
    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X: np.float array of shape (num_features,)
                The instance to fit.
            y: int (Default=None)
                The label of the instance (Optional for unsupervised models)
        """
        pass

    @abstractmethod
    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X: np.float array of shape (num_features,)
                The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            score: float
                The anomalousness score of the input instance.
        """
        pass

    def fit_score_partial(self, X, y=None):
        """Applies fit_partial and score_partial to the next instance, respectively.

        Args:
            X: np.float array of shape (num_features,)
                The instance to fit and score.
            y: int (Default=None)
                The label of the instance (Optional for unsupervised models)

        Returns:
            score: float
                The anomalousness score of the input instance.
        """
        return self.fit_partial(X, y).score_partial(X)

    def fit(self, X, y=None):
        """Fits the model to all instances in order.

        Args:
            X: np.float array of shape (num_instances, num_features)
                The instances in order to fit.
            y: np.int array of shape (num_instances, )  (Default=None)
                The labels of the instances in order to fit (Optional for unsupervised models)
        Returns:
            self: object
                Fitted model.
        """
        for xi, yi in _iterate(X, y):
            self.fit_partial(xi, yi)

        return self

    def score(self, X):
        """Scores all instaces via score_partial iteratively.

        Args:
            X: np.float array of shape (num_instances, num_features)
                The instances in order to score.

        Returns:
            scores: np.float array of shape (num_instances,)
                The anomalousness scores of the instances in order.
        """
        for xi, _ in _iterate(X):
            yield self.score_partial(xi)

    def fit_score(self, X, y=None):
        """This helper method applies fit_score_partial to all instances in order.

        Args:
            X: np.float array of shape (num_instances, num_features)
                The instances in order to fit.
            y: np.int array of shape (num_instances, )  (Default=None)
                The labels of the instances in order to fit (Optional for unsupervised models)
        Returns:
            scores: np.float array of shape (num_instances,)
                The anomalousness scores of the instances in order.
        """
        for xi, yi in _iterate(X, y):
            yield self.fit_score_partial(xi, yi)
