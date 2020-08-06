from pysad.core.base_model import BaseModel


class KitNet(BaseModel):
    """KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders :cite:`mirsky2018kitsune`. This model directly uses the implementation from `KitNET-py <https://github.com/ymirsky/KitNET-py>`_.

    Args:
        num_features: int
            The number of features in your input dataset.
        max_size_ae: int (Default=10)
            The maximum size of any autoencoder in the ensemble layer.
        grace_feature_mapping: int (Default=None)
            The number of instances the network will learn from before producing anomaly scores.
        grace_anomaly_detector: int (Default=50000)
            The number of instances which will be taken to learn the feature mapping. If 'None', then FM_grace_period=AM_grace_period
        learning_rate: float (Default=0.1)
            The default stochastic gradient descent learning rate for all autoencoders in the KitNET instance.
        hidden_ratio: float (Default=0.75)
            the default ratio of hidden to visible neurons. E.g., 0.75 will cause roughly a 25% compression in the hidden layer.
    """
    def __init__(self, max_size_ae=10, grace_feature_mapping=None, grace_anomaly_detector=50000, learning_rate=0.1, hidden_ratio=0.75):
        self.grace_feature_mapping = grace_feature_mapping
        self.hidden_ratio = hidden_ratio
        self.learning_rate = learning_rate
        self.max_size_ae = max_size_ae
        self.grace_anomaly_detector = grace_anomaly_detector
        self.to_init = True
        from pysad.models.kitnet_model import KitNET as kit


    def fit_partial(self, X, y=None):
        """Fits the model to next instance. Simply, adds the instance to the window.

        Args:
            X: np.float array of shape (num_features,)
                The instance to fit.
            y: int (Default=None)
                Ignored since the model is unsupervised.

        Returns:
            self: object
                Returns the self.
        """
        if self.to_init:
            self.model = kit.KitNET(self.num_features, self.max_size_ae, self.grace_feature_mapping, self.grace_anomaly_detector,
                                    self.learning_rate, self.hidden_ratio)
            self.to_init = False
        self.model.train(X)

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X: np.float array of shape (num_features,)
                The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            score: float
                The anomalousness score of the input instance.
        """
        return self.model.execute(X)
