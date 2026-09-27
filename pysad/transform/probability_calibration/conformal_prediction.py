from pysad.core.base_postprocessor import BasePostprocessor
import numpy as np

from pysad.utils.window import Window, UnlimitedWindow


class ConformalProbabilityCalibrator(BasePostprocessor):
    """This class provides an interface to convert the scores into probabilities through conformal prediction. Note that :cite:`laxhammar2013online` fits conformal calibration to already fitted samples' scores by the model whereas :cite:`ishimtsev2017conformal` fits the conformal calibration to some window of previous samples that are just before the target instance.
    This calibrator transforms a score into a conformal p-value: the fraction of scores in the window that are greater than or equal to the target score. Low values mean anomalous, e.g. alert when the calibrated score is below 0.05.
    The target score is assumed to be fitted before it is transformed (e.g. via `fit_transform_partial`), so it is counted in the window. For a score that is not in the window, the exact p-value would be (count + 1) / (n + 1).

        Args:
            windowed (bool): Whether the probability calibrator is windowed so that forget scores that are older than `window_size`.
            window_size (int): The number of scores kept in the window. Must be at least 1 when `windowed` is True; ignored otherwise.

        Raises:
            ValueError: If windowed is True and window_size is less than 1.
    """

    def __init__(self, windowed=True, window_size=300):
        self.windowed = windowed
        self.window_size = window_size
        self.window = Window(window_size=self.window_size) if self.windowed else UnlimitedWindow()

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
            float: Conformal p-value of the score.
        """
        return (np.sum(np.array(self.window.get()) >= score)) / (len(self.window.get()))
