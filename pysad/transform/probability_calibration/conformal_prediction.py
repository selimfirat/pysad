from pysad.core.base_postprocessor import BasePostprocessor
import numpy as np

from pysad.utils.window import Window, UnlimitedWindow


class ConformalProbabilityCalibrator(BasePostprocessor):
    """This class provides an interface to convert the scores into probabilities through conformal prediction. Note that :cite:`laxhammar2013online` fits conformal calibration to already fitted samples' scores by the model whereas :cite:`ishimtsev2017conformal` fits the conformal calibration to some window of previous samples that are just before the target instance.
    This calibrator transforms by providing target score divided by the number of instances that are fitted before to this calibrator as transformation result.

        Args:
            windowed (bool): Whether the probability calibrator is windowed so that forget scores that are older than `window_size`.
            window_size (int): The size of window for running average and std. Ignored if `running_statistics` parameter is False.
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
            float: Processed score.
        """
        return (np.sum(np.array(self.window.get()) > score)) / (len(self.window.get()))
