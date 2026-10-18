from pysad.core.base_model import BaseModel
from pysad.models.kitnet_model import KitNET as kit


class KitNet(BaseModel):
    """Scores each instance by the reconstruction error of an online ensemble of small autoencoders (KitNET) :cite:`mirsky2018kitsune`.

    This model uses the implementation from `KitNET-py <https://github.com/ymirsky/KitNET-py>`_ with two changes: the 0-1 normalization only shifts an input that was constant in training, which KitNET-py divides by 1e-16, and the feature mapping learns from ``grace_feature_mapping`` instances instead of one more.

    As in the paper, the model learns only during two grace periods at the start of the stream and then only scores. The first ``grace_feature_mapping`` instances learn which features go to which autoencoder. KitNET-py scores them 0. This model scores them with temporary autoencoders on the feature mapping so far, which it builds again each time the number of instances doubles (1, 2, 4, 8 and so on). After the feature mapping, it builds the autoencoders again on the final mapping, so the model after the grace periods is the same as without the temporary autoencoders. The next ``grace_anomaly_detector`` instances train the autoencoders, and their scores come from autoencoders that are still training. Choose both so that their sum is a small part of the stream and still covers its normal behavior. The defaults, 100 and 1,000 instances, keep the 1:10 ratio of the KitNET-py example (5,000 and 50,000 on a capture of 100,000 packets), scaled down so that a stream of a few hundred instances gets real scores. For a long stream whose start is free of anomalies, such as the paper's network captures (it trains on the first million packets), longer grace periods give the autoencoders more normal behavior to learn.

    Args:
        max_size_ae (int): The maximum size of any autoencoder in the ensemble layer (Default=10).
        grace_feature_mapping (int): The number of instances used to learn the feature mapping. If ``None``, the feature mapping and anomaly detector grace periods are equal (Default=100).
        grace_anomaly_detector (int): The number of instances used to train the autoencoders after the feature mapping is learned (Default=1000).
        learning_rate (float): The default stochastic gradient descent learning rate for all autoencoders in the KitNET instance (Default=0.1).
        hidden_ratio (float): The default ratio of hidden to visible neurons. E.g., 0.75 will cause roughly a 25% compression in the hidden layer (Default=0.75).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the initial weights of the autoencoders, which all share it. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self,
        max_size_ae=10,
        grace_feature_mapping=100,
        grace_anomaly_detector=1000,
        learning_rate=0.1,
        hidden_ratio=0.75,
        random_state=None,
    ):

        self.grace_feature_mapping = grace_feature_mapping
        self.hidden_ratio = hidden_ratio
        self.learning_rate = learning_rate
        self.max_size_ae = max_size_ae
        self.grace_anomaly_detector = grace_anomaly_detector
        self.random_state = random_state
        self.to_init = True

    def fit_partial(self, X, y=None):
        """Fits the model to the next instance.

        The first ``grace_feature_mapping`` instances learn the feature mapping;
        the next ``grace_anomaly_detector`` instances train the autoencoders.
        Before the first instance, :meth:`score_partial` returns ``0.0``.
        During both grace periods, it returns real scores from temporary or
        partially trained autoencoders, so they are unreliable for alerting.
        After both grace periods the model stops learning and only scores.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        if self.to_init:
            self.num_features = X.shape[0]
            self.model = kit.KitNET(
                self.num_features,
                self.max_size_ae,
                self.grace_feature_mapping,
                self.grace_anomaly_detector,
                self.learning_rate,
                self.hidden_ratio,
                random_state=self.random_state,
            )
            self.to_init = False
        # After both grace periods KitNET only scores, which is score_partial's job.
        if self.model.inGrace():
            self.model.train(X)

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if self.to_init or self.model.outputLayer is None:
            # KitNET builds its first autoencoders on the first instance, and there is nothing to
            # score with before that.
            return 0.0
        else:
            return self.model.execute(X)
