from models.base_model import BaseModel
from models.reference_window_model import ReferenceWindowModel
from pyod.models.iforest import IForest

class IForestASD(ReferenceWindowModel):

    """

    Note that concept drift is not implemented since it is a part of the simulation. See Algorithm 2 in "An Anomaly Detection Approach Based on Isolation Forest Algorithm for Streaming Data using Sliding Window" paper.

    Reference: An Anomaly Detection Approach Based on Isolation Forest Algorithm for Streaming Data using Sliding Window
    """
    def __init__(self, initial_window_X=None, window_size=2048, **kwargs):
        super().__init__(IForest, window_size, window_size, initial_window_X, **kwargs)

