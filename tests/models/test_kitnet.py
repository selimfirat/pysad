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

    assert model.feature_map_ == [[0]]
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
    # Temporary autoencoders score the feature mapping instances, so only the first one scores 0.
    assert scores[0] == 0.0
    assert np.isfinite(scores).all()
    assert (scores[1:] > 0.0).all()


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
    assert model.model.fm.n_updates == 5
    assert model.model.temporary

    model.fit(X[5:])
    assert model.model.fm.n_updates == 5
    assert not model.model.temporary
    assert model.model.output_layer.n_trained == 7


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

    from pysad.models.kitnet import _CorClust

    rng = np.random.RandomState(seed)
    X = rng.rand(30, 60) @ rng.rand(60, 60)
    clusterer = _CorClust(X.shape[1])
    for x in X:
        clusterer.update(x)
    D = clusterer.corr_dist()
    dendro = to_tree(linkage(D[np.triu_indices(X.shape[1], 1)]))

    assert clusterer.cluster(max_size) == _break_recursively(dendro, max_size)


def test_feature_mapping_of_a_wide_stream_does_not_recurse():
    """Regression test for #311: constant features are all at distance 1, so single
    linkage chains them into a dendrogram deeper than Python's recursion limit, which the
    recursive split hit."""
    X = np.ones((10, 2000))

    model = KitNet(grace_feature_mapping=5, grace_anomaly_detector=5, random_state=0).fit(X)

    assert sorted(i for cluster in model.feature_map_ for i in cluster) == list(range(2000))
    assert all(len(cluster) <= 10 for cluster in model.feature_map_)


def test_default_grace_periods_score_a_short_stream():
    """Regression test for #312: the defaults gave 0 for the first 50,000
    instances, so every score of a stream shorter than that was 0."""
    model = KitNet(random_state=0)
    scores = model.fit_score(_stream(num_instances=1500))

    # Only the first instance has no autoencoders to score it.
    assert scores[0] == 0.0
    assert (scores[1:] > 0.0).all()
    assert not model.model.in_grace()


def test_temporary_autoencoders_do_not_change_the_final_model():
    """During the feature mapping, temporary autoencoders are built at 1, 2, 4, ... instances.
    After it, the autoencoders are built again on the final mapping and train on the same
    instances as a KitNET that is given that mapping, so the two score the same."""
    from pysad.models.kitnet import _KitNET

    X = _stream(num_instances=30)
    model = _KitNET(3, 2, fm_grace_period=8, ad_grace_period=8, random_state=0)
    built = []
    create = model._create_autoencoders
    model._create_autoencoders = lambda m, temporary: (
        built.append(model.n_trained + 1),
        create(m, temporary),
    )
    for x in X[:16]:
        model.train(x)
    given_map = _KitNET(
        3, 2, fm_grace_period=0, ad_grace_period=8, feature_map=model.feature_map, random_state=0
    )
    for x in X[8:16]:
        given_map.train(x)

    assert built == [1, 2, 4, 8, 9]
    assert not model.temporary
    for x in X[16:]:
        assert model.execute(x) == given_map.execute(x)


def test_kitnet_explain_returns_per_feature_weights():
    import numpy as np

    from pysad.models import KitNet

    X = np.random.RandomState(0).rand(200, 4)
    model = KitNet(grace_feature_mapping=10, grace_anomaly_detector=10, random_state=0).fit(X[:100])

    weights = model.explain_partial(X[101])

    assert weights.shape == (4,)
    assert np.all(weights >= 0.0)
    assert abs(weights.sum() - 1.0) < 1e-12
    assert weights.dtype == np.float64


def test_kitnet_explain_does_not_change_state():
    import numpy as np

    from pysad.models import KitNet

    X = np.random.RandomState(0).rand(200, 4)
    model = KitNet(grace_feature_mapping=10, grace_anomaly_detector=10, random_state=0).fit(X[:100])

    before = model.score_partial(X[101])
    model.explain_partial(X[101])
    after = model.score_partial(X[101])

    assert before == after


def test_kitnet_explain_is_reproducible():
    import numpy as np

    from pysad.models import KitNet

    X = np.random.RandomState(0).rand(200, 4)

    model_a = KitNet(grace_feature_mapping=10, grace_anomaly_detector=10, random_state=0)
    model_a.fit(X[:100])
    weights_a = model_a.explain_partial(X[101])

    model_b = KitNet(grace_feature_mapping=10, grace_anomaly_detector=10, random_state=0)
    model_b.fit(X[:100])
    weights_b = model_b.explain_partial(X[101])

    np.testing.assert_array_equal(weights_a, weights_b)


def test_kitnet_explain_weights_depend_on_the_instance():
    import numpy as np

    from pysad.models import KitNet

    X = np.random.RandomState(0).rand(200, 4)
    model = KitNet(grace_feature_mapping=10, grace_anomaly_detector=10, random_state=0).fit(X[:100])

    normal = model.explain_partial(X[101])
    outside = model.explain_partial(np.array([100.0, -50.0, 25.0, -200.0]))

    assert not np.allclose(normal, outside)


def test_kitnet_explain_returns_zeros_before_any_fit():
    import numpy as np

    from pysad.models import KitNet

    model = KitNet(grace_feature_mapping=5, grace_anomaly_detector=5, random_state=0)

    weights = model.explain_partial(np.array([0.3, 0.6, 0.1]))

    assert weights.shape == (3,)
    assert np.all(weights == 0.0)


def test_kitnet_explain_rejects_nan():
    import numpy as np

    from pysad.models import KitNet

    X = np.random.RandomState(0).rand(200, 4)
    model = KitNet(grace_feature_mapping=10, grace_anomaly_detector=10, random_state=0).fit(X[:100])

    anomalous = X[101].copy()
    anomalous[1] = np.nan
    with pytest.raises(ValueError, match="KitNet does not accept NaN values"):
        model.explain_partial(anomalous)
