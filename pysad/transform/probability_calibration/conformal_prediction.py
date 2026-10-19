from pysad.core.base_postprocessor import BasePostprocessor
from pysad.utils.window import Window


class ConformalProbabilityCalibrator(BasePostprocessor):
    """Converts each score into the fraction of scores in the window that are lower than it, by conformal prediction :cite:`laxhammar2013online,ishimtsev2017conformal`.

    The calibrator works on the scores that the model gave when each instance arrived. This follows the sliding calibration queue of Ishimtsev et al. (2017) (LDCD), where the window holds the scores that the model gave when each instance arrived, with the test score included. When ``window_size`` is the queue length plus one, the calibrator returns one minus that conformal p-value. In contrast, Conformal Anomaly Detection (CAD) :cite:`laxhammar2013online` recomputes the nonconformity scores of all previous instances against the current training set for each new instance.

    This calibrator transforms a score into the fraction of scores in the window that are lower than it, i.e. one minus its conformal p-value. As for model scores, higher values mean more anomalous, e.g. alert when the calibrated score is above 0.95, which is a conformal p-value below 0.05.

    The target score is assumed to be fitted before it is transformed (e.g. via ``fit_transform_partial``), so it is counted in the window. For a score that is not in the window, the exact conformal p-value would be (count + 1) / (n + 1), where count is the number of scores in the window that are greater than or equal to it.

    Args:
        window_size (int): The number of scores kept in the window. Must be at least 1 (Default=300).

    Raises:
        ValueError: If window_size is less than 1.
    """

    def __init__(self, window_size=300):
        self.window_size = window_size
        self.window = Window(window_size=self.window_size)

    def fit_partial(self, score):
        """Fits particular (next) timestep's score to train the postprocessor.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.window.update(score)

        return self

    def transform_partial(self, score):
        """Transforms given score.

        Args:
            score (float): Input score.

        Returns:
            float: One minus the conformal p-value of the score.
        """
        scores = self.window.get()
        return sum(1 for s in scores if s < score) / len(scores)
