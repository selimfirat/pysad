from abc import ABC, abstractmethod
from pysad.utils import _iterate


class BaseScoreEnsembler(ABC):
    """Abstract base class for the scoring based ensemble anomaly detection methods.
    """

    @abstractmethod
    def transform_partial(self, scores):
        """Fits ensembler to the list of scores for the next timestep.

        Args:
            scores: numpy array of type np.float32 and shape (num_models,)
                List of scores from various anomaly detectors.

        Returns:
            score: float
                Resulting anomaly score.
        """
        pass

    def transform(self, all_scores):
        """Fits ensembler to the list of scores.

        Args:
            all_scores: numpy array of shape (num_timesteps, num_models)
                List of scores for all timesteps.

        Returns:
            self: object
                Returns fitted ensembler.

        """
        for xi, scores in _iterate(all_scores):
            self.transform_partial(scores)

        return self
