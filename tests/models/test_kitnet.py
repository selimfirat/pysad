import numpy as np
import pytest

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


def test_fit_partial_does_not_score_after_the_grace_periods():
    """fit_partial used to run the autoencoders on every instance after the grace periods and
    discard the score, so each instance went through them twice with fit_score_partial."""
    model = KitNet(grace_feature_mapping=5, grace_anomaly_detector=5, random_state=0).fit(_stream())

    assert model.model.n_executed == 0


def test_scores_start_once_the_autoencoders_train():
    """Regression test for #226: the instance that ended the feature mapping was scored by
    autoencoders that had not trained yet, whose ranges for 0-1 normalization were still
    (inf, -inf), so its score was nan."""
    scores = KitNet(grace_feature_mapping=5, grace_anomaly_detector=5, random_state=0).fit_score(
        _stream()
    )

    # The feature mapper learns from the first 6 instances, and the autoencoders from the next.
    np.testing.assert_array_equal(scores[:6], 0.0)
    assert np.isfinite(scores).all()
    assert (scores[6:] > 0.0).all()


def test_inputs_constant_in_training_are_not_scaled_by_1e16():
    """Regression test for #226: 0-1 normalization divided by max - min + 1e-16, so an input whose
    range was still zero was scaled by 1e16 as soon as it changed. That hit every input of the
    output layer after its first instance, and a feature that was constant in training."""
    X = _stream()
    X[:, 1] = 0.0
    model = KitNet(grace_feature_mapping=5, grace_anomaly_detector=5, random_state=0)
    scores = model.fit_score(X)
    x = X[-1].copy()
    x[1] = 0.001

    assert scores.max() < 10.0
    assert model.score_partial(x) == pytest.approx(scores[-1], rel=0.05)
