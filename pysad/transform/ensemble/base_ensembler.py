from abc import ABC

from pysad.core.base_transformer import BaseTransformer


class BaseScoreEnsembler(BaseTransformer,metaclass=ABC):
    """Abstract base class for the scoring based ensemble anomaly detection methods.
    """
    pass
