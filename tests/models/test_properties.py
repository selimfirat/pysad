"""Property-based tests for the guarantees every model makes, whatever the data.

Each test draws a stream with Hypothesis, including the edge cases that fixed test data rarely
covers (a single feature, constant features, duplicate instances, very short streams), and checks
one guarantee for every model in ``pysad.models.__all__``:

- every instance gets a finite ``float`` score;
- the batch methods (``fit``, ``score``, ``fit_score``) agree with the per-instance methods;
- pickling a model mid-stream does not change the scores that follow;
- no method modifies its input.

A model that raises on any of the drawn streams fails the test as well.
"""

import copy
import pickle

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import pysad.models
from pysad.utils import fix_seed

SEED = 61
MAX_FEATURES = 4
# KNNCAD's shortest probationary period is 48, and the streams must outlast it.
MAX_STREAM_LENGTH = 70
# Examples per model and property. CI runs the suite on one core in places, so keep it small;
# raise it locally to search further.
MAX_EXAMPLES = 10

UNIVARIATE_MODELS = {
    "KNNCAD",
    "MedianAbsoluteDeviation",
    "RelativeEntropy",
    "SeasonalESD",
    "SeasonalHybridESD",
    "StandardAbsoluteDeviation",
}

# Values the streams are drawn from. Small pools make constant features and duplicate instances
# likely; the float range covers values far from 0 and from each other.
FINITE_VALUES = st.one_of(
    st.sampled_from([0.0, 1.0, -1.0, 0.5]),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)

PROPERTIES = settings(
    max_examples=MAX_EXAMPLES,
    # The first call of a model (e.g. Inqmad's jitted functions) can take seconds.
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    # The same examples on every run, so a failure in CI reproduces locally.
    derandomize=True,
)

# Open bugs that a property finds, by test and model. Each is a strict xfail, so the test starts
# failing once the bug is fixed, as a reminder to remove it here.
KITNET_ONE_FEATURE = "#225: KitNet crashes on a stream with one feature."
KNOWN_FAILURES = {
    "test_scores_are_finite_floats": {
        "KitNet": f"{KITNET_ONE_FEATURE} #226: it scores nan before its autoencoders train.",
        "StandardAbsoluteDeviation": "#224: VarianceMeter goes negative on a constant stream.",
    },
    "test_batch_methods_agree_with_partial_methods": {"KitNet": KITNET_ONE_FEATURE},
    "test_pickling_mid_stream_keeps_later_scores": {
        "KitNet": f"{KITNET_ONE_FEATURE} #227: unpickling unties its decoder weights.",
    },
    "test_inputs_are_not_modified": {"KitNet": KITNET_ONE_FEATURE},
}
# Known failures that do not depend on the data, where shrinking the failing example only costs
# time. They are tracked by a strict xfail elsewhere.
SKIPPED = {
    "test_pickling_mid_stream_keeps_later_scores": {
        "Inqmad": "#159: Inqmad cannot be pickled at all; test_pickling.py xfails it.",
    },
}


def _model_params(model_name, num_features, data):
    """The constructor arguments of `model_name`, sized to `num_features` and short streams."""
    if model_name == "LocalOutlierProbability":
        initial_X = data.draw(
            _streams(num_features, min_length=12, max_length=20), label="initial_X"
        )
        return {"initial_X": initial_X, "num_neighbors": 10}

    ones, zeros = [1.0] * num_features, [0.0] * num_features
    params = {
        "ExactStorm": {"window_size": 20, "max_radius": 0.1},
        "HalfSpaceTrees": {
            "feature_mins": np.array(zeros),
            "feature_maxes": np.array(ones),
            "window_size": 10,
            "num_trees": 5,
            "max_depth": 5,
        },
        "IForestASD": {"window_size": 20, "n_estimators": 10},
        "Inqmad": {"input_shape": num_features, "dim_x": 16, "gamma": 1.0, "batch_size": 8},
        "KitNet": {"grace_feature_mapping": 10, "grace_anomaly_detector": 10},
        "KNNCAD": {"probationary_period": 48},
        "LODA": {"num_bins": 5, "num_random_cuts": 10},
        "RelativeEntropy": {"min_val": 0.0, "max_val": 1.0, "window_size": 5, "num_bins": 5},
        "RobustRandomCutForest": {"num_trees": 3, "shingle_size": 2, "tree_size": 16},
        "RSHash": {
            "feature_mins": np.array(zeros),
            "feature_maxes": np.array(ones),
            "num_components": 10,
        },
        "SeasonalESD": {"period": 2, "window_size": 8, "max_anomalies": 2},
        "SeasonalHybridESD": {"period": 2, "window_size": 8, "max_anomalies": 2},
        "xStream": {"num_components": 10, "n_chains": 10, "depth": 5, "window_size": 10},
    }

    return params.get(model_name, {})


@st.composite
def _streams(draw, num_features, min_length=1, max_length=MAX_STREAM_LENGTH):
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
    else:
        num_features = data.draw(st.integers(1, MAX_FEATURES), label="num_features")

    X = data.draw(_streams(num_features, min_length=min_length), label="X")
    y = data.draw(hnp.arrays(np.int64, X.shape[0], elements=st.integers(0, 1)), label="y")
    params = _model_params(model_name, num_features, data)

    return X, y, params


def _make_model(model_name, params):
    fix_seed(SEED)
    return getattr(pysad.models, model_name)(**copy.deepcopy(params))


def _fit_score_partial(model, X, y):
    # Models such as RobustRandomCutForest and RandomModel draw from the global numpy generator.
    fix_seed(SEED + 1)
    return np.array([model.fit_score_partial(xi, yi) for xi, yi in zip(X, y, strict=True)])


def _models(test_name):
    known_failures = KNOWN_FAILURES.get(test_name, {})
    skipped = SKIPPED.get(test_name, {})
    params = []
    for name in pysad.models.__all__:
        marks = []
        if name in known_failures:
            marks.append(pytest.mark.xfail(reason=known_failures[name], strict=True))
        if name in skipped:
            marks.append(pytest.mark.skip(reason=skipped[name]))
        params.append(pytest.param(name, marks=marks))

    return params


@pytest.mark.parametrize("model_name", _models("test_scores_are_finite_floats"))
@PROPERTIES
@given(data=st.data())
def test_scores_are_finite_floats(model_name, data):
    X, y, params = _draw_case(model_name, data)

    # Scored while fitting, and scored after fitting the whole stream.
    model = _make_model(model_name, params)
    scores = [model.fit_score_partial(xi, yi) for xi, yi in zip(X, y, strict=True)]
    model = _make_model(model_name, params)
    for xi, yi in zip(X, y, strict=True):
        model.fit_partial(xi, yi)
    scores += [model.score_partial(xi) for xi in X]

    assert all(type(score) is float for score in scores)
    assert np.isfinite(scores).all(), scores


@pytest.mark.parametrize("model_name", _models("test_batch_methods_agree_with_partial_methods"))
@PROPERTIES
@given(data=st.data())
def test_batch_methods_agree_with_partial_methods(model_name, data):
    X, y, params = _draw_case(model_name, data)

    streamed = _fit_score_partial(_make_model(model_name, params), X, y)
    model = _make_model(model_name, params)
    fix_seed(SEED + 1)
    batched = model.fit_score(X, y)
    np.testing.assert_array_equal(batched, streamed)

    model = _make_model(model_name, params)
    fix_seed(SEED + 1)
    for xi, yi in zip(X, y, strict=True):
        model.fit_partial(xi, yi)
    streamed = np.array([model.score_partial(xi) for xi in X])

    model = _make_model(model_name, params)
    fix_seed(SEED + 1)
    batched = model.fit(X, y).score(X)
    np.testing.assert_array_equal(batched, streamed)


@pytest.mark.parametrize("model_name", _models("test_pickling_mid_stream_keeps_later_scores"))
@PROPERTIES
@given(data=st.data())
def test_pickling_mid_stream_keeps_later_scores(model_name, data):
    X, y, params = _draw_case(model_name, data, min_length=2)
    split = data.draw(st.integers(1, X.shape[0] - 1), label="split")

    model = _make_model(model_name, params)
    _fit_score_partial(model, X[:split], y[:split])
    restored = pickle.loads(pickle.dumps(model))

    expected = _fit_score_partial(model, X[split:], y[split:])
    actual = _fit_score_partial(restored, X[split:], y[split:])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("model_name", _models("test_inputs_are_not_modified"))
@PROPERTIES
@given(data=st.data())
def test_inputs_are_not_modified(model_name, data):
    X, y, params = _draw_case(model_name, data)
    X_before, y_before, params_before = X.copy(), y.copy(), copy.deepcopy(params)

    # Not _make_model, which passes a copy of the parameters.
    fix_seed(SEED)
    model = getattr(pysad.models, model_name)(**params)
    model.fit(X, y)
    model.score(X)
    model.fit_score(X, y)
    for xi, yi in zip(X, y, strict=True):
        model.fit_partial(xi, yi)
        model.score_partial(xi)
        model.fit_score_partial(xi, yi)

    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)
    for name, value in params_before.items():
        np.testing.assert_array_equal(params[name], value, err_msg=name)
