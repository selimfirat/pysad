"""Property-based tests for the guarantees every model makes, whatever the data.

Each test draws a stream with Hypothesis, including the edge cases that fixed test data rarely
covers (a single feature, constant features, duplicate instances, very short streams), and checks
one guarantee for every model in ``pysad.models.__all__``:

- every instance gets a finite ``float`` score;
- the batch methods (``fit``, ``score``, ``fit_score``) agree with the per-instance methods;
- no method modifies its input;
- reusing one input array for every instance gives the same scores as passing separate arrays.

A model that raises on any of the drawn streams fails the test as well.
"""

import copy

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import pysad.models
from pysad.utils import fix_seed
from tests.models.conftest import (
    EDGE_MODELS,
    TIMESTAMPED_MODELS,
    UNIVARIATE_MODELS,
    active_model_names,
    short_stream_params,
)

SEED = 61
MAX_FEATURES = 4
# Examples per model and property. CI runs the suite on one core in places, so keep it small;
# raise it locally to search further.
MAX_EXAMPLES = 5


def _max_stream_length(model_name):
    # KNNCAD's shortest probationary period is 48, and DAMP uses longer derandomized streams;
    # other models use short test windows <= 20 and MiLOF summarizes and merges by length 24.
    return 70 if model_name in ("KNNCAD", "DAMP") else 28


# Values the streams are drawn from. Small pools make constant features and duplicate instances
# likely; the float range covers values far from 0 and from each other.
FINITE_VALUES = st.one_of(
    st.sampled_from([0.0, 1.0, -1.0, 0.5]),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)

PROPERTIES = settings(
    max_examples=MAX_EXAMPLES,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
    # The same examples on every run, so a failure in CI reproduces locally.
    derandomize=True,
)

# Open bugs that a property finds, by test and model. Each is a strict xfail, so the test starts
# failing once the bug is fixed, as a reminder to remove it here.
KNOWN_FAILURES = {}
# Known failures that do not depend on the data, where shrinking the failing example only costs
# time. They are tracked by a strict xfail elsewhere.
SKIPPED = {}


def _model_params(model_name, num_features, data):
    """The constructor arguments of `model_name`, sized to `num_features` and Hypothesis streams."""
    if model_name == "LocalOutlierProbability":
        initial_X = data.draw(
            _streams(num_features, min_length=12, max_length=20), label="initial_X"
        )
        return {"initial_X": initial_X, "num_neighbors": 10}

    # Hypothesis-specific overrides for extreme values (-1e6) and very short (<=28) streams.
    params = {
        "AnyOut": {
            "outlier_score": "density",
            "max_height": 2,
            "decay": 0.1,
            "snapshot_interval": 5,
        },
        # A short probationary period, so that the streams reach the anomaly likelihood.
        "HTM": {
            "min_val": 0.0,
            "max_val": 1.0,
            "probationary_period": 8,
            "num_columns": 16,
            "num_cells_per_column": 2,
        },
        # The drawn values reach -1e6, below the domain of log(1 + x).
        "MStream": {"log_transform": False},
        "OeSNNUAD": {"window_size": 5, "num_output_neurons": 5},
    }

    return params.get(model_name) or short_stream_params(model_name, num_features, seed=SEED)


@st.composite
def _streams(draw, num_features, min_length=1, max_length=28):
    """A stream of shape (num_instances, num_features) with constant features and repeats."""
    num_instances = draw(st.integers(min_length, max_length))
    X = draw(hnp.arrays(np.float64, (num_instances, num_features), elements=FINITE_VALUES))

    for j in range(num_features):
        if draw(st.booleans()):
            X[:, j] = X[0, j]

    if num_instances > 1:
        for i in draw(st.lists(st.integers(1, num_instances - 1), max_size=5)):
            X[i] = X[draw(st.integers(0, i - 1))]

    return X


def _draw_case(model_name, data, min_length=1):
    """Draws a stream and labels for `model_name`, and returns them with its constructor args."""
    if model_name in UNIVARIATE_MODELS:
        num_features = 1
    elif model_name in TIMESTAMPED_MODELS or model_name in EDGE_MODELS:
        num_features = 2
    else:
        num_features = data.draw(st.integers(1, MAX_FEATURES), label="num_features")

    X = data.draw(
        _streams(num_features, min_length=min_length, max_length=_max_stream_length(model_name)),
        label="X",
    )
    y = data.draw(hnp.arrays(np.int64, X.shape[0], elements=st.integers(0, 1)), label="y")
    params = _model_params(model_name, num_features, data)

    return X, y, params


def _make_model(model_name, params):
    fix_seed(SEED)
    return getattr(pysad.models, model_name)(**copy.deepcopy(params))


def _models(test_name):
    known_failures = KNOWN_FAILURES.get(test_name, {})
    skipped = SKIPPED.get(test_name, {})
    params = []
    for name in active_model_names():
        marks = []
        if name in known_failures:
            marks.append(pytest.mark.xfail(reason=known_failures[name], strict=True))
        if name in skipped:
            marks.append(pytest.mark.skip(reason=skipped[name]))
        params.append(pytest.param(name, marks=marks))

    return params


@pytest.mark.parametrize("model_name", _models("test_streaming_model_invariants"))
@PROPERTIES
@given(data=st.data())
def test_streaming_model_invariants(model_name, data):
    X, y, params = _draw_case(model_name, data)
    X_before, y_before = X.copy(), y.copy()
    seq_params = copy.deepcopy(params)
    array_params = {k: v.copy() for k, v in seq_params.items() if isinstance(v, np.ndarray)}

    # 1. Partial fit_score yields finite Python floats and leaves inputs unmodified.
    fix_seed(SEED)
    seq_model = getattr(pysad.models, model_name)(**seq_params)
    fix_seed(SEED + 1)
    partial_fit_scores = [seq_model.fit_score_partial(xi, yi) for xi, yi in zip(X, y, strict=True)]
    assert all(type(score) is float for score in partial_fit_scores)
    streamed = np.array(partial_fit_scores)
    assert np.isfinite(streamed).all(), streamed
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)
    for name, value in array_params.items():
        np.testing.assert_array_equal(seq_params[name], value, err_msg=name)

    # 2. Batch fit_score agrees with sequential fit_score_partial and leaves inputs unchanged.
    batch_model = _make_model(model_name, params)
    fix_seed(SEED + 1)
    batched = batch_model.fit_score(X, y)
    np.testing.assert_array_equal(batched, streamed)
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)

    # 3. Sequential fit_partial + score_partial produces finite Python floats and matches batch fit().score().
    fit_then_score_model = _make_model(model_name, params)
    fix_seed(SEED + 1)
    for xi, yi in zip(X, y, strict=True):
        fit_then_score_model.fit_partial(xi, yi)
    post_scores = [fit_then_score_model.score_partial(xi) for xi in X]
    assert all(type(score) is float for score in post_scores)
    post_arr = np.array(post_scores)
    assert np.isfinite(post_arr).all(), post_arr

    batch_fit_score_model = _make_model(model_name, params)
    fix_seed(SEED + 1)
    batched_post = batch_fit_score_model.fit(X, y).score(X)
    np.testing.assert_array_equal(batched_post, post_arr)

    # 4. Reusing one input buffer gives identical scores and fit/score leaves slice inputs unchanged.
    buf_model = _make_model(model_name, params)
    buffer = np.empty_like(X[0])
    fix_seed(SEED + 1)
    buffered = []
    for xi, yi in zip(X, y, strict=True):
        buffer[:] = xi
        buffered.append(buf_model.fit_score_partial(buffer, yi))
    np.testing.assert_array_equal(buffered, streamed)

    indices = sorted({*range(min(2, len(X))), *range(max(0, len(X) - 2), len(X))})
    for idx in indices:
        xi, yi = X[idx], y[idx]
        buf_model.fit_partial(xi, yi)
        buf_model.score_partial(xi)
        buf_model.fit_score_partial(xi, yi)
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)
