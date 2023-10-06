from abc import ABC, abstractmethod
import numpy as np
from pysad.utils import _iterate


class BasePostprocessor(ABC):
    """Base class for postprocessing methods.
    """

    @abstractmethod
    def fit_partial(self, score):
        """Fits particular (next) timestep's score to train the postprocessor.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        pass

    @abstractmethod
    def transform_partial(self, score):
        """Transforms given score.

        Args:
            score (float): Input score.

        Returns:
            float: Processed score.
        """
        pass

    def fit_transform_partial(self, score):
        """Shortcut method that iteratively applies fit_partial and transform_partial, respectively.

        Args:
            score (float): Input score.

        Returns:
            float: Processed score.
        """
        return self.fit_partial(score).transform_partial(score)

    def transform(self, scores):
        """Shortcut method that iteratively applies transform_partial to all instances in order.

        Args:
            np.float64 array of shape (num_instances,): Input scores.

        Returns:
            np.float64 array of shape (num_instances,): Processed scores.
        """
        processed_scores = np.empty(scores.shape[0], dtype=np.float64)
        for i, (score, _) in enumerate(_iterate(scores)):
            processed_scores[i] = self.transform_partial(score)

        return processed_scores

    def fit(self, scores):
        """Shortcut method that iteratively applies fit_partial to all instances in order.

        Args:
            np.float64 array of shape (num_instances,): Input scores.

        Returns:
            object: self.
        """
        for i, (score, _) in enumerate(_iterate(scores)):
            self.fit_partial(score)

        return self

    def fit_transform(self, scores):
        """Shortcut method that iteratively applies fit_transform_partial to all instances in order.

        Args:
            np.float64 array of shape (num_instances,): Input scores.

        Returns:
            np.float64 array of shape (num_instances,): Processed scores.
        """
        processed_scores = np.empty(scores.shape[0], dtype=np.float64)
        for i, (score, _) in enumerate(_iterate(scores)):
            processed_scores[i] = self.fit_transform_partial(score)

        return processed_scores
