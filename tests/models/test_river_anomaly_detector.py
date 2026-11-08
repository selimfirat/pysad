import pickle
from functools import partial

import numpy as np
import pytest

pytest.importorskip("river")


def test_scores_match_pysad_before_learning():
    from pysad.integrations import RiverAnomalyDetector
    from pysad.models import StandardAbsoluteDeviation

    direct = StandardAbsoluteDeviation()
    detector = RiverAnomalyDetector(StandardAbsoluteDeviation)
    wrapped = detector.model_
    for value in [1.0, 3.0, 2.0, 10.0, -5.0]:
        expected = direct.fit_score_partial(np.array([value]))
        score = detector.score_one({"value": value})
        assert type(score) is float
        assert score == expected
        assert detector.score_one({"value": value}) == score
        detector.learn_one({"value": value})
        assert wrapped.score_partial(np.array([value])) == direct.score_partial(np.array([value]))


@pytest.mark.parametrize("first_method", ["score_one", "learn_one"])
@pytest.mark.parametrize("feature_names", [None, ("b", 1)])
def test_dictionary_order_does_not_change_feature_positions(first_method, feature_names):
    from pysad.integrations import RiverAnomalyDetector
    from pysad.models import LODA

    detector = RiverAnomalyDetector(partial(LODA, random_state=42), feature_names=feature_names)
    direct = LODA(random_state=42)
    getattr(detector, first_method)({"b": 2.0, 1: 10.0})
    if first_method == "learn_one":
        direct.fit_partial(np.array([2.0, 10.0]))

    for b, a in [(4.0, 11.0), (1.0, 13.0), (10.0, 20.0)]:
        x = {1: a, "b": b}
        assert detector.score_one(x) == direct.score_partial(np.array([b, a]))
        detector.learn_one(x)
        direct.fit_partial(np.array([b, a]))


def test_explicit_order_matches_model_columns():
    from pysad.integrations import RiverAnomalyDetector
    from pysad.models import LODA

    model = LODA(random_state=42).fit(np.array([[2.0, 10.0], [4.0, 11.0], [1.0, 13.0]]))
    detector = RiverAnomalyDetector(partial(LODA, random_state=42), feature_names=("b", "a"))
    for b, a in [(2.0, 10.0), (4.0, 11.0), (1.0, 13.0)]:
        detector.learn_one({"a": a, "b": b})
    assert detector.score_one({"a": 20.0, "b": 10.0}) == model.score_partial(np.array([10.0, 20.0]))


@pytest.mark.parametrize("method", ["score_one", "learn_one"])
@pytest.mark.parametrize("x", [{"a": 2.0}, {"a": 2.0, "c": 1.0}, {"a": 2.0, "b": 1.0, "c": 3.0}])
def test_changed_feature_sets_are_rejected(method, x):
    from pysad.integrations import RiverAnomalyDetector
    from pysad.models import LODA

    detector = RiverAnomalyDetector(partial(LODA, random_state=42))
    detector.learn_one({"a": 1.0, "b": 2.0})
    original = pickle.loads(pickle.dumps(detector))
    with pytest.raises(ValueError, match="feature"):
        getattr(detector, method)(x)
    assert detector.score_one({"a": 3.0, "b": 4.0}) == original.score_one({"a": 3.0, "b": 4.0})


def test_duplicate_feature_names_are_rejected():
    from pysad.integrations import RiverAnomalyDetector
    from pysad.models import NullModel

    with pytest.raises(ValueError, match="feature_names"):
        RiverAnomalyDetector(NullModel, feature_names=("a", "a"))


def test_river_pipeline_scores_before_learning_and_survives_pickling():
    from river import base, preprocessing

    from pysad.integrations import RiverAnomalyDetector
    from pysad.models import StandardAbsoluteDeviation

    detector = RiverAnomalyDetector(StandardAbsoluteDeviation)
    assert isinstance(detector, base.AnomalyDetector)
    pipeline = preprocessing.StandardScaler() | detector
    direct = StandardAbsoluteDeviation()
    scaler = preprocessing.StandardScaler()
    for value in [1.0, 3.0, 2.0, 10.0]:
        x = {"value": value}
        assert pipeline.score_one(x) == direct.score_partial(
            np.array([scaler.transform_one(x)["value"]])
        )
        scaler.learn_one(x)
        direct.fit_partial(np.array([scaler.transform_one(x)["value"]]))
        pipeline.learn_one(x)

    restored = pickle.loads(pickle.dumps(pipeline))
    for value in [20.0, -8.0]:
        x = {"value": value}
        assert restored.score_one(x) == pipeline.score_one(x)
        restored.learn_one(x)
        pipeline.learn_one(x)


@pytest.mark.parametrize("pipeline", [False, True])
def test_default_clone_is_untrained(pipeline):
    from river import preprocessing

    from pysad.integrations import RiverAnomalyDetector
    from pysad.models import StandardAbsoluteDeviation

    detector = RiverAnomalyDetector(StandardAbsoluteDeviation)
    fresh = RiverAnomalyDetector(StandardAbsoluteDeviation)
    if pipeline:
        detector = preprocessing.StandardScaler() | detector
        fresh = preprocessing.StandardScaler() | fresh
    for value in [1.0, 3.0, 2.0]:
        detector.learn_one({"value": value})

    cloned = detector.clone()
    assert detector.score_one({"value": 10.0}) != 0.0
    for value in [10.0, -5.0, 4.0]:
        x = {"renamed": value}
        assert cloned.score_one(x) == fresh.score_one(x)
        cloned.learn_one(x)
        fresh.learn_one(x)


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("feature_names", [None, ("b", "a")])
def test_default_clone_resets_loda_with_configured_parameters(pipeline, feature_names):
    from river import preprocessing

    from pysad.integrations import RiverAnomalyDetector
    from pysad.models import LODA

    factory = partial(LODA, num_bins=6, num_random_cuts=11, random_state=42)
    detector = RiverAnomalyDetector(factory, feature_names=feature_names)
    fresh = RiverAnomalyDetector(factory, feature_names=feature_names)
    if pipeline:
        detector = preprocessing.StandardScaler() | detector
        fresh = preprocessing.StandardScaler() | fresh
    for b, a in [(2.0, 10.0), (4.0, 11.0), (1.0, 13.0)]:
        detector.learn_one({"b": b, "a": a})

    cloned = detector.clone()
    for b, a in [(10.0, 20.0), (-5.0, 3.0), (0.0, 50.0)]:
        x = {"a": a, "b": b}
        assert cloned.score_one(x) == fresh.score_one(x)
        cloned.learn_one(x)
        fresh.learn_one(x)
    assert cloned.score_one({"b": 7.0, "a": 25.0}) == fresh.score_one({"b": 7.0, "a": 25.0})


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("feature_names", [None, ("b", "a")])
def test_clone_with_attributes_preserves_scores_and_feature_order(pipeline, feature_names):
    from river import preprocessing

    from pysad.integrations import RiverAnomalyDetector
    from pysad.models import LODA

    detector = RiverAnomalyDetector(partial(LODA, random_state=42), feature_names=feature_names)
    if pipeline:
        detector = preprocessing.StandardScaler() | detector
    for b, a in [(2.0, 10.0), (4.0, 11.0), (1.0, 13.0)]:
        detector.learn_one({"b": b, "a": a})

    cloned = detector.clone(include_attributes=True)
    control = pickle.loads(pickle.dumps(detector))
    for b, a in [(10.0, 20.0), (-5.0, 3.0)]:
        x = {"a": a, "b": b}
        assert cloned.score_one(x) == detector.score_one(x)
        cloned.learn_one(x)
        assert detector.score_one(x) == control.score_one(x)
        detector.learn_one(x)
        control.learn_one(x)
