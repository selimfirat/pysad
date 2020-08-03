from pysad.models.reference_window_model import ReferenceWindowModel
from pyod.models.iforest import IForest


class IForestASD(ReferenceWindowModel):
    """An Anomaly Detection Approach Based on Isolation Forest Algorithm for Streaming Data using Sliding Window :cite:`ding2013anomaly`. Note that concept drift is not implemented since it is a part of the simulation. See Algorithm 2 in "An Anomaly Detection Approach Based on Isolation Forest Algorithm for Streaming Data using Sliding Window" paper. This method is unsupervised so it is not needed to give y as parameter.

    Args:
        window_size: int (Default=2048)
            The size of the reference window and its sliding.
        initial_window_X: np.float array of shape (num_initial_instances,num_features)
            The initial window to fit for initial calibration period. We simply apply fit to these instances.
    """

    def __init__(self, initial_window_X=None, window_size=2048):
        super().__init__(IForest, window_size, window_size, initial_window_X)
        # TODO: implement concept drit method

