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

    # The feature mapper learns from the first 5 instances, and the autoencoders from the next.
    # Each instance is scored before the model learns it, so the sixth one still scores 0.
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

    # The autoencoders score the seventh instance after they trained on one instance, so its
    # score is high, but far below the 1e16 scale of the bug.
    assert scores.max() < 1e3
    assert model.score_partial(x) == pytest.approx(scores[-1], rel=0.05)


def test_grace_periods_last_the_given_numbers_of_instances():
    """The feature mapping used to learn from grace_feature_mapping + 1 instances."""
    X = _stream()
    model = KitNet(grace_feature_mapping=5, grace_anomaly_detector=7, random_state=0)

    model.fit(X[:5])
    assert model.model.FM.N == 5
    assert model.model.outputLayer is None

    model.fit(X[5:])
    assert model.model.FM.N == 5
    assert model.model.outputLayer.n == 7


def _break_recursively(dendro, max_size):
    """The recursive dendrogram split of KitNET-py's corClust."""
    if dendro.count <= max_size:
        return [dendro.pre_order()]
    return _break_recursively(dendro.get_left(), max_size) + _break_recursively(
        dendro.get_right(), max_size
    )


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("max_size", [1, 3, 10, 50])
def test_feature_mapping_matches_the_recursive_split(seed, max_size):
    from scipy.cluster.hierarchy import linkage, to_tree

    from pysad.models.kitnet_model.corClust import corClust

    rng = np.random.RandomState(seed)
    X = rng.rand(30, 60) @ rng.rand(60, 60)
    clusterer = corClust(X.shape[1])
    for x in X:
        clusterer.update(x)
    D = clusterer.corrDist()
    dendro = to_tree(linkage(D[np.triu_indices(X.shape[1], 1)]))

    assert clusterer.cluster(max_size) == _break_recursively(dendro, max_size)


def test_feature_mapping_of_a_wide_stream_does_not_recurse():
    """Regression test for #311: constant features are all at distance 1, so single
    linkage chains them into a dendrogram deeper than Python's recursion limit, which the
    recursive split hit."""
    X = np.ones((10, 2000))

    model = KitNet(grace_feature_mapping=5, grace_anomaly_detector=5, random_state=0).fit(X)

    assert sorted(i for cluster in model.model.v for i in cluster) == list(range(2000))
    assert all(len(cluster) <= 10 for cluster in model.model.v)


def test_default_grace_periods_score_a_short_stream():
    """Regression test for #312: the defaults gave 0 for the first 50,000
    instances, so every score of a stream shorter than that was 0."""
    model = KitNet(random_state=0)
    scores = model.fit_score(_stream(num_instances=1500))

    # The first 100 instances train the feature mapper, and the next one builds the autoencoders.
    # Each instance is scored before the model learns it, so these 101 instances score 0.
    np.testing.assert_array_equal(scores[:101], 0.0)
    assert (scores[101:] > 0.0).all()
    assert not model.model.inGrace()
