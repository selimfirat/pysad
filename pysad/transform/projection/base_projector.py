from pysad.core.base_transformer import BaseTransformer


class BaseProjector(BaseTransformer):

    def __init__(self, num_components):
        """Abstract base class for online projection methods.

        Args:
            num_components: The number of dimensions that the target will be projected into.
        """
        self.num_components = num_components
