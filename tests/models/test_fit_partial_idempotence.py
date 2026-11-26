"""Property-based tests for fit_partial idempotence across all detectors.

Fitting the same instance twice must not change the anomaly score that the model
assigns to that instance:

    fit_partial(x); score_partial(x)  ==  fit_partial(x); fit_partial(x); score_partial(x)

This idempotence property holds for detectors whose streaming update rule does not
change the score of an already-fitted instance when it is fitted again. It does not
hold for detectors whose update rule accumulates per-instance statistics
(for example xStream, D3, ApproxStorm) or resubsamples memory at random
(SmartSifter): fitting the same instance a second time changes their state and
therefore its score.

Those detectors are tracked as strict xfails. A detector that becomes idempotent
makes its strict xfail start failing, which is the intended signal to remove it.
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
MAX_EXAMPLES = 8

PROPERTIES = settings(
    max_examples=MAX_EXAMPLES,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=True,
)

# Values the streams are drawn from. Small pools make constant features and duplicate
# instances likely; the float range covers values far from 0 and from each other.
FINITE_VALUES = st.one_of(
    st.sampled_from([0.0, 1.0, -1.0, 0.5]),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)

# Detectors whose fit_partial is non-idempotent by design (accumulating or stochastic
# update rule changes state when the same instance is fitted twice). Each is a strict
# xfail, so the test starts failing if the detector becomes idempotent.
NON_IDEMPOTENT_MODELS = frozenset(
    {
        "ApproxStorm",
        "CostSensitiveStreamModel",
        "D3",
        "ModelEnsemble",
        "MStream",
        "OnlineOneClassSVM",
        "RSHash",
        "SmartSifter",
        "StreamingAutoregressiveDetector",
        "StrictModelEnsemble",
        "ThreshLEAP",
        "xStream",
    }
)


def _max_stream_length(model_name):
    # KNNCAD's shortest probationary period is 48, and DAMP uses longer derandomized
    # streams; other models use short test windows <= 20.
    return 70 if model_name in ("KNNCAD", "DAMP") else 28


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


def _model_params(model_name, num_features, data):
    """The constructor arguments of `model_name`, sized to `num_features` and Hypothesis streams."""
    params = short_stream_params(model_name, num_features, seed=SEED)
    # Hypothesis streams reach -1e6, which is outside the domain of log(1 + x).
    if model_name == "MStream":
        params["log_transform"] = False
    # Smaller quantum matrices stay numerically stable on extreme drawn values.
    if model_name == "Inqmad":
        params["dim_x"] = 8
        params["initial_size"] = 8
    return params


def _draw_case(model_name, data, min_length=1):
    """Draws a stream and constructor args for `model_name`."""
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
    params = _model_params(model_name, num_features, data)
    return X, params


def _make_model(model_name, params):
    fix_seed(SEED)
    return getattr(pysad.models, model_name)(**copy.deepcopy(params))


def _models(test_name):
    params = []
    for name in active_model_names():
        marks = []
        if name in NON_IDEMPOTENT_MODELS:
            marks.append(
                pytest.mark.xfail(
                    reason=f"{name} fit_partial is non-idempotent by design", strict=True
                )
            )
        params.append(pytest.param(name, marks=marks))
    return params


@pytest.mark.parametrize("model_name", _models("test_fit_partial_idempotence"))
@PROPERTIES
@given(data=st.data())
def test_fit_partial_idempotence(model_name, data):
    """Fitting the same instance twice yields the same score as fitting it once."""
    X, params = _draw_case(model_name, data)
    idx = data.draw(st.integers(0, len(X) - 1), label="instance_index")
    xi = X[idx]
    yi = data.draw(st.integers(0, 1), label="label")

    # Fit the instance once, then score it.
    once_model = _make_model(model_name, params)
    once_model.fit_partial(xi, yi)
    score_once = once_model.score_partial(xi)

    # Fit the same instance twice, then score it.
    twice_model = _make_model(model_name, params)
    twice_model.fit_partial(xi, yi)
    twice_model.fit_partial(xi, yi)
    score_twice = twice_model.score_partial(xi)

    assert score_once == pytest.approx(score_twice, rel=1e-12, abs=1e-12), (
        f"{model_name}: fit_partial is not idempotent: "
        f"score once={score_once}, score twice={score_twice}"
    )
