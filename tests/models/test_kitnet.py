import numpy as np

from pysad.models import KitNet


def _stream(num_instances=40, num_features=3):
    return np.random.RandomState(0).rand(num_instances, num_features)


def test_one_feature_stream_maps_to_one_autoencoder():
    """Regression test for #225: the feature mapping used to pass scipy's linkage an empty
    distance matrix for a single feature."""
    X = _stream(num_features=1)

    model = KitNet(grace_feature_mapping=5, grace_anomaly_detector=5, random_state=0).fit(X)

    assert model.model.v == [[0]]
    assert np.isfinite(model.score(X)).all()


def test_does_not_print(capsys):
    KitNet(grace_feature_mapping=5, grace_anomaly_detector=5, random_state=0).fit(_stream())

    assert capsys.readouterr().out == ""
