"""Every model scores constant, zero-variance, and single-instance streams without failing.

A stream of identical instances (constant), a stream where each feature is constant but the
features differ (zero variance), and a stream with one instance are the inputs that most
often hit a division by zero or an empty window in a detector. These tests pin the guarantee
for every model in ``pysad.models.__all__``: every public entry point returns a finite score
of the right shape, the per-instance methods agree with the batch methods, decision-capable
models return 0 or 1, and no NumPy floating-point error is raised.
"""

import numpy as np
import pytest

import pysad.models
from pysad.utils import fix_seed
from tests.models.conftest import active_model_names, num_features_for, short_stream_params

NUM_INSTANCES = 12
ACTIVE_MODELS = list(active_model_names())


def _make_model(name):
    fix_seed(61)

    return getattr(pysad.models, name)(**short_stream_params(name, num_features_for(name)))


def _labels(name, num_instances):
    if name == "PerfectModel":
        return np.zeros(num_instances, dtype=np.int32)
    return None


def _constant_stream(num_features):
    """A stream where every instance is the same vector."""
    return np.full((NUM_INSTANCES, num_features), 0.5)


def _zero_variance_stream(num_features):
    """A stream where every feature is constant and the features differ from each other."""
    row = np.arange(1, num_features + 1, dtype=np.float64) * 0.25

    return np.tile(row, (NUM_INSTANCES, 1))


def _single_instance(num_features):
    """A stream with one constant instance."""
    return np.full((1, num_features), 0.5)


EDGE_STREAMS = {
    "constant": _constant_stream,
    "zero_variance": _zero_variance_stream,
    "single_instance": _single_instance,
}


def _decision_model_names():
    return [
        name
        for name in ACTIVE_MODELS
        if getattr(pysad.models, name)._output_kind == "decision"
        or getattr(pysad.models, name)._has_decision_rule
    ]


DECISION_MODELS = _decision_model_names()


@pytest.mark.parametrize("stream_name", list(EDGE_STREAMS))
@pytest.mark.parametrize("name", ACTIVE_MODELS, ids=ACTIVE_MODELS)
def test_fit_score_edge_stream(name, stream_name):
    X = EDGE_STREAMS[stream_name](num_features_for(name))
    y = _labels(name, len(X))

    model = _make_model(name)
    fix_seed(62)
    with np.errstate(all="raise"):
        scores = model.fit_score(X, y)

    assert scores.shape == (len(X),)
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("stream_name", list(EDGE_STREAMS))
@pytest.mark.parametrize("name", ACTIVE_MODELS, ids=ACTIVE_MODELS)
def test_partial_methods_match_batch_on_edge_stream(name, stream_name):
    X = EDGE_STREAMS[stream_name](num_features_for(name))
    y = _labels(name, len(X))

    model = _make_model(name)
    fix_seed(62)
    with np.errstate(all="raise"):
        if y is None:
            partial_scores = [model.fit_score_partial(xi) for xi in X]
        else:
            partial_scores = [model.fit_score_partial(xi, yi) for xi, yi in zip(X, y, strict=True)]

    assert all(type(score) is float for score in partial_scores)
    assert np.all(np.isfinite(partial_scores))

    batch_model = _make_model(name)
    fix_seed(62)
    with np.errstate(all="raise"):
        batch_scores = batch_model.fit_score(X, y)
    np.testing.assert_array_equal(batch_scores, np.array(partial_scores))


@pytest.mark.parametrize("name", ACTIVE_MODELS, ids=ACTIVE_MODELS)
def test_score_partial_single_instance_on_unfitted_model(name):
    num_features = num_features_for(name)
    model = _make_model(name)

    with np.errstate(all="raise"):
        score = model.score_partial(np.full(num_features, 0.5))

    assert type(score) is float
    assert np.isfinite(score)


@pytest.mark.parametrize("stream_name", list(EDGE_STREAMS))
@pytest.mark.parametrize("name", DECISION_MODELS, ids=DECISION_MODELS)
def test_predict_on_edge_stream(name, stream_name):
    X = EDGE_STREAMS[stream_name](num_features_for(name))
    y = _labels(name, len(X))

    model = _make_model(name)
    fix_seed(62)
    model.fit(X, y)
    with np.errstate(all="raise"):
        predictions = model.predict(X)
        partial_prediction = model.predict_partial(X[0])

    assert predictions.shape == (len(X),)
    assert set(np.unique(predictions)) <= {0, 1}
    assert partial_prediction in (0, 1)
