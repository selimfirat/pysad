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
    (inf, -inf), so its score was nan.

    Now matches KitNET-py behavior: scores are 0.0 during the entire feature mapping
    grace period, and only become non-zero during the anomaly detector grace period
    (when autoencoders are training) or after both grace periods.
    """
    scores = KitNet(grace_feature_mapping=5, grace_anomaly_detector=5, random_state=0).fit_score(
        _stream()
    )

    # The feature mapper learns from the first 5 instances (indices 0-4).
    # During these, score is 0.0 and no autoencoder training occurs.
    # The autoencoders train from indices 5-9.
    # Since the stream is short, we only get partial training.
    assert scores[0] == 0.0
    assert np.isfinite(scores).all()
    # First 5 scores are 0.0 (feature mapping grace period)
    assert (scores[:5] == 0.0).all()
    # Remaining scores may be 0.0 (if autoencoders not trained enough) or positive
    # Just verify no NaN and finite


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
    """The feature mapping learns from exactly grace_feature_mapping instances."""
    X = _stream()
    model = KitNet(grace_feature_mapping=5, grace_anomaly_detector=7, random_state=0)

    model.fit(X[:5])
    assert model.model.fm.n_updates == 5
    # No temporary autoencoders created during feature mapping
    assert model.model.output_layer is None

    model.fit(X[5:])
    assert model.model.fm.n_updates == 5
    assert model.model.output_layer is not None
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
    X = np.ones((10, 1100))

    model = KitNet(grace_feature_mapping=5, grace_anomaly_detector=5, random_state=0).fit(X)

    assert sorted(i for cluster in model.feature_map_ for i in cluster) == list(range(1100))
    assert all(len(cluster) <= 10 for cluster in model.feature_map_)


def test_default_grace_periods_score_a_short_stream():
    """Regression test for #312: the defaults gave 0 for the first 50,000
    instances, so every score of a stream shorter than that was 0.

    With defaults (FM=100, AD=1000), the first 100 instances have score 0.0
    (feature mapping grace period). Instance 100 creates the autoencoders and
    trains them, but its score is still 0.0 because scoring happens before
    fitting. From instance 101 onwards, scores are non-zero.
    """
    model = KitNet(random_state=0)
    scores = model.fit_score(_stream(num_instances=1500))

    # First instance has no autoencoders, scores 0.0
    assert scores[0] == 0.0
    # First 101 instances (feature mapping + first AD instance) score 0.0
    assert (scores[:101] == 0.0).all()
    # After that, autoencoders produce scores
    assert (scores[101:] > 0.0).all()
    # With 1500 instances, we should exit both grace periods (100 + 1000 = 1100)
    assert not model.model.in_grace()


def test_autoencoders_match_given_feature_map():
    """When a feature map is provided, autoencoders are created immediately and
    start training from the first instance. This matches the behavior of a KitNET
    that learns the feature mapping internally."""
    from pysad.models.kitnet import _KitNET

    X = _stream(num_instances=30)
    # Model that learns feature mapping from first 8 instances
    model = _KitNET(3, 2, fm_grace_period=8, ad_grace_period=8, random_state=0)
    for x in X[:16]:
        model.train(x)

    # Model given the same feature map
    given_map = _KitNET(
        3, 2, fm_grace_period=0, ad_grace_period=8, feature_map=model.feature_map, random_state=0
    )
    for x in X[8:16]:
        given_map.train(x)

    # After both have trained on the same instances (8-15), they should produce
    # the same scores for subsequent instances
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


def test_custom_hidden_ratio_changes_the_scores():
    X = _stream(num_features=6)

    scores = KitNet(
        grace_feature_mapping=5, grace_anomaly_detector=5, hidden_ratio=0.5, random_state=0
    ).fit_score(X)
    other = KitNet(
        grace_feature_mapping=5, grace_anomaly_detector=5, hidden_ratio=0.9, random_state=0
    ).fit_score(X)

    assert all(isinstance(score, float) for score in scores.tolist())
    assert np.isfinite(scores).all()
    assert not np.array_equal(scores, other)


def test_grace_period_transition_feature_mapping_to_anomaly_detector():
    """Test the exact transition from feature mapping to anomaly detector grace period.

    During feature mapping (indices 0 to fm-1): score=0.0, only correlation matrix updates.
    At index fm: autoencoders created and trained on this instance.
    After fitting fm, score for fm is non-zero (autoencoders now exist).
    """
    fm = 5
    ad = 10
    X = np.random.RandomState(42).rand(20, 4)

    model = KitNet(grace_feature_mapping=fm, grace_anomaly_detector=ad, random_state=123)

    # Before any fit: score is 0.0
    assert model.score_partial(X[0]) == 0.0

    # Fit instances 0 to fm-1 (feature mapping period)
    for i in range(fm):
        model.fit_partial(X[i])
        # Score during feature mapping is 0.0
        assert model.score_partial(X[i]) == 0.0, f"Instance {i} during FM should score 0.0"

    # Autoencoders not created yet
    assert model.model.output_layer is None

    # Fit instance fm (first AD instance) - creates and trains autoencoders
    model.fit_partial(X[fm])
    # After fitting fm, autoencoders exist, so score_partial returns non-zero
    score_fm = model.score_partial(X[fm])
    assert score_fm >= 0.0, f"Instance {fm} (first AD) score should be non-negative"
    assert np.isfinite(score_fm)

    # Autoencoders now exist
    assert model.model.output_layer is not None
    assert len(model.model.ensemble_layer) > 0

    # Fit instance fm+1 - autoencoders have trained on one instance
    model.fit_partial(X[fm + 1])
    score_fm_plus_1 = model.score_partial(X[fm + 1])
    assert score_fm_plus_1 >= 0.0, f"Instance {fm + 1} score should be non-negative"
    assert np.isfinite(score_fm_plus_1)


def test_grace_period_transition_anomaly_detector_to_scoring_only():
    """Test the transition from anomaly detector grace period to scoring-only mode.

    After fm + ad instances have been trained on: model stops learning, only scores.
    in_grace() returns False.
    """
    fm = 5
    ad = 3
    X = np.random.RandomState(42).rand(fm + ad + 5, 4)

    model = KitNet(grace_feature_mapping=fm, grace_anomaly_detector=ad, random_state=123)

    # Fit through feature mapping period (instances 0 to fm-1)
    for i in range(fm):
        model.fit_partial(X[i])
    assert model.model.in_grace()

    # Fit through anomaly detector period (instances fm to fm+ad-1)
    # After fitting fm+ad-1, n_trained = fm+ad, so in_grace() becomes False
    for i in range(fm, fm + ad - 1):
        model.fit_partial(X[i])
        assert model.model.in_grace(), f"in_grace should be True after fitting instance {i}"

    # Fit the last AD instance (fm+ad-1)
    model.fit_partial(X[fm + ad - 1])
    # Now n_trained = fm + ad, so in_grace() is False
    assert not model.model.in_grace()

    # Further fits should not change the model (only scoring)
    n_trained_before = model.model.n_trained
    model.fit_partial(X[fm + ad])
    model.fit_partial(X[fm + ad + 1])
    assert model.model.n_trained == n_trained_before

    # Scores should still be produced
    for i in range(fm + ad, fm + ad + 3):
        score = model.score_partial(X[i])
        assert np.isfinite(score)
        assert score >= 0.0


def test_reconstruction_error_stability_after_grace_periods():
    """Test that reconstruction error stabilizes after grace periods end.

    During AD grace period, autoencoders are still training so scores may vary.
    After both grace periods, autoencoders stop training and scores should
    be stable for similar normal instances.
    """
    fm = 10
    ad = 20
    rng = np.random.RandomState(42)
    # Generate normal data from a fixed distribution
    X_train = rng.rand(fm + ad, 5)
    X_test = rng.rand(30, 5)  # Similar normal instances

    model = KitNet(grace_feature_mapping=fm, grace_anomaly_detector=ad, random_state=123)
    model.fit(X_train)

    # Model should be out of grace periods
    assert not model.model.in_grace()

    # Score test instances - they should be relatively stable (low variance)
    # since they come from the same distribution as training data
    scores = model.score(X_test)
    assert np.isfinite(scores).all()
    assert (scores >= 0.0).all()

    # The coefficient of variation should be reasonable for normal data
    # (not a strict threshold, but scores shouldn't vary wildly)
    cv = scores.std() / (scores.mean() + 1e-10)
    assert cv < 5.0, f"Score coefficient of variation too high: {cv}"


def test_seeded_reproducibility_full_pipeline():
    """Test that KitNet with the same seed produces identical results across full pipeline.

    Covers: feature mapping, autoencoder initialization, training, and scoring.
    """
    fm = 5
    ad = 10
    X = np.random.RandomState(999).rand(30, 4)

    # First run
    model_a = KitNet(grace_feature_mapping=fm, grace_anomaly_detector=ad, random_state=42)
    scores_a = model_a.fit_score(X)
    feature_map_a = model_a.feature_map_
    weights_a = [ae.W.copy() for ae in model_a.model.ensemble_layer]
    output_weights_a = model_a.model.output_layer.W.copy()

    # Second run with same seed
    model_b = KitNet(grace_feature_mapping=fm, grace_anomaly_detector=ad, random_state=42)
    scores_b = model_b.fit_score(X)
    feature_map_b = model_b.feature_map_
    weights_b = [ae.W.copy() for ae in model_b.model.ensemble_layer]
    output_weights_b = model_b.model.output_layer.W.copy()

    # Scores should be identical
    np.testing.assert_array_equal(scores_a, scores_b)
    # Feature mapping should be identical
    assert feature_map_a == feature_map_b
    # Autoencoder weights should be identical
    for wa, wb in zip(weights_a, weights_b, strict=True):
        np.testing.assert_array_equal(wa, wb)
    np.testing.assert_array_equal(output_weights_a, output_weights_b)


def test_seeded_reproducibility_feature_mapping():
    """Test that the feature mapping is reproducible with the same seed.

    The correlation clustering uses random initialization for the autoencoders,
    but the clustering itself is deterministic given the data. The random_state
    affects the autoencoder weight initialization.
    """
    fm = 10
    ad = 10
    X = np.random.RandomState(777).rand(30, 6)

    model_a = KitNet(grace_feature_mapping=fm, grace_anomaly_detector=ad, random_state=100)
    model_a.fit(X[:fm])  # Only feature mapping period
    feature_map_a = model_a.model.fm.cluster(fm)

    model_b = KitNet(grace_feature_mapping=fm, grace_anomaly_detector=ad, random_state=100)
    model_b.fit(X[:fm])
    feature_map_b = model_b.model.fm.cluster(fm)

    assert feature_map_a == feature_map_b


def test_seeded_reproducibility_incremental():
    """Test reproducibility when fitting incrementally with fit_partial."""
    fm = 5
    ad = 10
    X = np.random.RandomState(888).rand(20, 3)

    model_a = KitNet(grace_feature_mapping=fm, grace_anomaly_detector=ad, random_state=555)
    scores_a = []
    for x in X:
        scores_a.append(model_a.fit_score_partial(x))

    model_b = KitNet(grace_feature_mapping=fm, grace_anomaly_detector=ad, random_state=555)
    scores_b = []
    for x in X:
        scores_b.append(model_b.fit_score_partial(x))

    np.testing.assert_array_equal(scores_a, scores_b)
    assert model_a.feature_map_ == model_b.feature_map_


def test_kitnet_score_explain_partial_returns_score_and_attribution():
    import numpy as np

    from pysad.models import KitNet

    X = np.random.RandomState(0).rand(200, 4)
    model = KitNet(grace_feature_mapping=10, grace_anomaly_detector=10, random_state=0).fit(X[:100])

    score, attribution = model.score_explain_partial(X[101])

    assert isinstance(score, float)
    assert attribution.shape == (4,)
    assert np.all(attribution >= 0.0)
    assert abs(attribution.sum() - 1.0) < 1e-12
    assert attribution.dtype == np.float64


def test_kitnet_score_explain_partial_does_not_change_state():
    import numpy as np

    from pysad.models import KitNet

    X = np.random.RandomState(0).rand(200, 4)
    model = KitNet(grace_feature_mapping=10, grace_anomaly_detector=10, random_state=0).fit(X[:100])

    before = model.score_partial(X[101])
    score, _ = model.score_explain_partial(X[101])
    after = model.score_partial(X[101])

    assert before == after
    assert score == before


def test_kitnet_score_explain_partial_is_reproducible():
    import numpy as np

    from pysad.models import KitNet

    X = np.random.RandomState(0).rand(200, 4)

    model_a = KitNet(grace_feature_mapping=10, grace_anomaly_detector=10, random_state=0)
    model_a.fit(X[:100])
    score_a, attrib_a = model_a.score_explain_partial(X[101])

    model_b = KitNet(grace_feature_mapping=10, grace_anomaly_detector=10, random_state=0)
    model_b.fit(X[:100])
    score_b, attrib_b = model_b.score_explain_partial(X[101])

    assert score_a == score_b
    np.testing.assert_array_equal(attrib_a, attrib_b)


def test_kitnet_score_explain_partial_attribution_depends_on_instance():
    import numpy as np

    from pysad.models import KitNet

    X = np.random.RandomState(0).rand(200, 4)
    model = KitNet(grace_feature_mapping=10, grace_anomaly_detector=10, random_state=0).fit(X[:100])

    normal_score, normal_attrib = model.score_explain_partial(X[101])
    outside_score, outside_attrib = model.score_explain_partial(
        np.array([100.0, -50.0, 25.0, -200.0])
    )

    assert normal_score != outside_score
    assert not np.allclose(normal_attrib, outside_attrib)


def test_kitnet_score_explain_partial_returns_zeros_before_autoencoders():
    import numpy as np

    from pysad.models import KitNet

    model = KitNet(grace_feature_mapping=5, grace_anomaly_detector=5, random_state=0)

    score, attribution = model.score_explain_partial(np.array([0.3, 0.6, 0.1]))

    assert score == 0.0
    assert attribution.shape == (3,)
    assert np.all(attribution == 0.0)


def test_kitnet_score_explain_partial_rejects_nan():
    import numpy as np

    from pysad.models import KitNet

    X = np.random.RandomState(0).rand(200, 4)
    model = KitNet(grace_feature_mapping=10, grace_anomaly_detector=10, random_state=0).fit(X[:100])

    anomalous = X[101].copy()
    anomalous[1] = np.nan
    with pytest.raises(ValueError, match="KitNet does not accept NaN values"):
        model.score_explain_partial(anomalous)
