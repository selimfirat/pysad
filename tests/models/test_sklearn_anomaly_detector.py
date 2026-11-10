import pickle

import numpy as np
import pytest


def test_score_direction_and_threshold_boundary():
    from pysad.integrations import SklearnAnomalyDetector
    from pysad.models import StandardAbsoluteDeviation

    model = StandardAbsoluteDeviation().fit(np.array([[0.0], [2.0]]))
    detector = SklearnAnomalyDetector(model, threshold=0.0)
    X = np.array([[1.0], [3.0], [-1.0]])
    scores = detector.score_samples(X)
    assert scores.dtype == np.float64
    np.testing.assert_allclose(scores, [0.0, -2.0, -2.0])
    np.testing.assert_allclose(detector.decision_function(X), [0.0, -2.0, -2.0])
    np.testing.assert_array_equal(detector.predict(X), [1, -1, -1])
    detector.set_params(threshold=3.0)
    np.testing.assert_allclose(detector.decision_function(X), [3.0, 1.0, 1.0])
    np.testing.assert_array_equal(detector.predict(X), [1, 1, 1])


@pytest.mark.parametrize("method", ["decision_function", "predict"])
def test_labels_require_an_explicit_threshold(method):
    from pysad.integrations import SklearnAnomalyDetector
    from pysad.models import StandardAbsoluteDeviation

    detector = SklearnAnomalyDetector(StandardAbsoluteDeviation().fit(np.array([[0.0], [2.0]])))
    np.testing.assert_allclose(detector.score_samples([[3.0]]), [-2.0])
    with pytest.raises(ValueError, match="threshold"):
        getattr(detector, method)([[3.0]])


@pytest.mark.parametrize("model_name", ["StandardAbsoluteDeviation", "LODA"])
def test_batch_operations_preserve_future_stream_behavior(model_name):
    from pysad import models
    from pysad.integrations import SklearnAnomalyDetector

    kwargs = {"random_state": 42} if model_name == "LODA" else {}
    model = getattr(models, model_name)(**kwargs).fit(np.array([[0.0], [2.0], [1.0]]))
    control = pickle.loads(pickle.dumps(model))
    detector = SklearnAnomalyDetector(model, threshold=2.0)
    X = np.array([[20.0], [-10.0], [0.5]])
    expected = -control.score(X)
    assert detector.fit(X) is detector
    for _ in range(2):
        np.testing.assert_array_equal(detector.score_samples(X), expected)
        detector.decision_function(X)
        detector.predict(X)
    for x in X:
        assert model.fit_score_partial(x) == control.fit_score_partial(x)


def test_fitted_pipeline_clone_and_pickle_preserve_scores():
    from sklearn.base import clone, is_outlier_detector
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.validation import check_is_fitted

    from pysad.integrations import SklearnAnomalyDetector
    from pysad.models import StandardAbsoluteDeviation

    history = np.array([[0.0], [2.0]])
    scaler = StandardScaler().fit(history)
    model = StandardAbsoluteDeviation().fit(scaler.transform(history))
    detector = SklearnAnomalyDetector(model, threshold=1.0)
    check_is_fitted(detector)
    assert is_outlier_detector(detector)
    pipeline = Pipeline([("scale", scaler), ("detect", detector)])
    X = np.array([[1.0], [5.0]])
    for candidate in [pipeline, clone(pipeline).fit(history), pickle.loads(pickle.dumps(pipeline))]:
        np.testing.assert_allclose(candidate.score_samples(X), [0.0, -4.0])
        np.testing.assert_array_equal(candidate.predict(X), [1, -1])
