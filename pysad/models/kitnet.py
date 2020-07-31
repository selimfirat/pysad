from pysad.models.base_model import BaseModel


class KitNet(BaseModel):

    def __init__(self, num_features, max_size_ae=10, grace_feature_mapping=5000, grace_anomaly_detector=50000, **kwargs):
        super().__init__(**kwargs)
        from pysad.models.kitnet_model import KitNET as kit

        self.model = kit.KitNET(num_features, max_size_ae, grace_feature_mapping, grace_anomaly_detector)

    def fit_partial(self, X, y=None):

        self.model.train(X)

        return self

    def score_partial(self, X):

        return self.model.execute(X)
