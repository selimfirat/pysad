from abc import ABC, abstractmethod

from pysad.utils import _iterate


class BasePostProcessor(ABC):
    """Base class for postprocessing methods.
    """

    @abstractmethod
    def fit_partial(self, score):
        """Fits particular (next) timestep's score to train the postprocessor.

        Args:
            score: float
                Input score.
        Returns:
            self: object
                The fitted postprocessor.
        """
        pass

    @abstractmethod
    def transform_partial(self, score):
        """Processes given score.

        Args:
            score: float
                Input score.

        Returns:
            postprocessed_score: float
                Postprocessed score.
        """
        pass

    def fit_transform_partial(self, score):
        """Shortcut method that iteratively applies fit_partial and transform_partial, respectively.

        Args:
            score: float
                Input score.

        Returns:
            postprocessed_score: float
                Postprocessed score.
        """
        self.fit_partial(score)

        return self.transform_partial(score)

    def transform(self, scores):
        """Shortcut method that iteratively applies transform_partial to all instances in order.

        Args:
            scores: np.float array of shape (num_instances,)
                Input scores.

        Returns:
            postprocessed_scores: np.float array of shape (num_instances,)
                Postprocessed scores.
        """
        for score, _ in _iterate(scores):
            yield self.transform_partial(score)

    def fit_transform(self, scores):
        """Shortcut method that iteratively applies fit_transform_partial to all instances in order.

        Args:
            scores: np.float array of shape (num_instances,)
                Input scores.

        Returns:
            postprocessed_scores: np.float array of shape (num_instances,)
                Postprocessed scores.
        """
        for score, _ in _iterate(scores):
            yield self.fit_transform_partial(score)
