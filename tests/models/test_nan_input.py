"""Every model rejects a NaN instance from every public entry point.

Before #402 the detectors handled NaN in different ways: four of them scored NaN
for the rest of the stream after one NaN instance, three raised three different
errors, and the rest returned a number that depended on how each one compared
NaN. ``BaseModel`` now raises a ``ValueError`` that names the model before any
of its methods run, so a rejected instance leaves the model unchanged and the
stream continues with finite scores. These tests pin that guarantee for every
model in ``pysad.models.__all__``.
"""

import numpy as np
import pytest

import pysad.models
from pysad.utils import fix_seed
from tests.models.conftest import active_model_names, num_features_for, short_stream_params

NUM_INSTANCES = 12
ACTIVE_MODELS = list(active_model_names())


def _num_features(name):
    return num_features_for(name)


def _make_model(name):
    fix_seed(61)

    return getattr(pysad.models, name)(**short_stream_params(name, _num_features(name)))


def _data(name, num_features):
    fix_seed(61)
    X = np.random.rand(NUM_INSTANCES, num_features)
    y = np.zeros(NUM_INSTANCES, dtype=np.int32) if name == "PerfectModel" else None

    return X, y


def _nan_instance(num_features):
    """An in-range instance whose first feature is NaN, so only NaN can be the reason for a raise."""
    instance = np.full(num_features, 0.5)
    instance[0] = np.nan

    return instance


@pytest.mark.parametrize("name", ACTIVE_MODELS, ids=ACTIVE_MODELS)
def test_partial_methods_reject_nan(name):
    num_features = _num_features(name)

    for method in ("fit_partial", "score_partial", "fit_score_partial"):
        model = _make_model(name)
        with pytest.raises(ValueError, match=f"{name} does not accept NaN values"):
            getattr(model, method)(_nan_instance(num_features))


@pytest.mark.parametrize("name", ACTIVE_MODELS, ids=ACTIVE_MODELS)
def test_batch_methods_reject_nan(name):
    num_features = _num_features(name)
    X, y = _data(name, num_features)
    X_nan_last = np.vstack([X, _nan_instance(num_features)])
    y_nan_last = None if y is None else np.append(y, np.int32(1))

    for method, args in (
        ("fit", (X_nan_last, y_nan_last)),
        ("score", (X_nan_last,)),
        ("fit_score", (X_nan_last, y_nan_last)),
    ):
        model = _make_model(name)
        # warm up on the clean part, so a model that needs state to score raises because of the NaN
        model.fit(X, y)
        with pytest.raises(ValueError, match=f"{name} does not accept NaN values"):
            getattr(model, method)(*args)


@pytest.mark.parametrize("name", ACTIVE_MODELS, ids=ACTIVE_MODELS)
def test_stream_continues_after_a_rejected_nan(name):
    """The issue's poisoning case: one NaN no longer silences the scores that follow."""
    num_features = _num_features(name)
    X, y = _data(name, num_features)
    model = _make_model(name)

    with pytest.raises(ValueError, match=f"{name} does not accept NaN values"):
        model.fit_score_partial(_nan_instance(num_features))

    scores = model.fit_score(X, y)

    assert scores.shape == (NUM_INSTANCES,)
    assert np.all(np.isfinite(scores))
