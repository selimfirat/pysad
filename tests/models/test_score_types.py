import numpy as np
import pytest

from pysad.utils import fix_seed

NUM_INSTANCES = 80
NUM_FEATURES = 3


def _initial_X(num_features):
    return np.random.RandomState(0).rand(25, num_features)


def _model_specs():
    from pyod.models.iforest import IForest
    from river import time_series

    from pysad import models
    from pysad.integrations import OneFitModel, PredictiveAnomalyDetector, ReferenceWindowModel
    from tests.models.conftest import (
        active_model_names,
        is_component_active,
        num_features_for,
        short_stream_params,
    )

    active = set(active_model_names())
    # (name, factory, number of features)
    specs = [
        (
            name,
            lambda name=name: getattr(models, name)(
                **short_stream_params(name, num_features_for(name, NUM_FEATURES))
            ),
            num_features_for(name, NUM_FEATURES),
        )
        for name in active_model_names()
    ]
    variants = [
        (
            "AnyOut",
            "AnyOut-density",
            lambda: models.AnyOut(max_height=2, outlier_score="density"),
            NUM_FEATURES,
        ),
        (
            "ChangeFinder",
            "ChangeFinder-outlier",
            lambda: models.ChangeFinder(change_point=False),
            1,
        ),
        (
            "HTM",
            "HTM-raw",
            lambda: models.HTM(min_val=0.0, max_val=1.0, likelihood=False, num_columns=64),
            2,
        ),
        (
            "MedianAbsoluteDeviation",
            "MedianAbsoluteDeviation-signed",
            lambda: models.MedianAbsoluteDeviation(absolute=False),
            1,
        ),
        (
            "StandardAbsoluteDeviation",
            "StandardAbsoluteDeviation-signed",
            lambda: models.StandardAbsoluteDeviation(absolute=False),
            1,
        ),
    ]
    specs += [(vid, fn, nf) for base_name, vid, fn, nf in variants if base_name in active]
    if is_component_active(OneFitModel):
        specs.append(
            (
                "OneFitModel",
                lambda: OneFitModel(IForest, _initial_X(NUM_FEATURES), n_estimators=10),
                NUM_FEATURES,
            )
        )
    if is_component_active(ReferenceWindowModel):
        specs.append(
            (
                "ReferenceWindowModel",
                lambda: ReferenceWindowModel(
                    IForest,
                    window_size=20,
                    sliding_size=10,
                    initial_window_X=_initial_X(NUM_FEATURES),
                    n_estimators=10,
                ),
                NUM_FEATURES,
            )
        )
    if is_component_active(PredictiveAnomalyDetector):
        specs.append(
            (
                "PredictiveAnomalyDetector",
                lambda: PredictiveAnomalyDetector(time_series.SNARIMAX(p=2, d=0, q=0)),
                1,
            )
        )

    return specs


MODEL_SPECS = _model_specs()


def _data(name, num_features):
    fix_seed(61)
    X = np.random.rand(NUM_INSTANCES, num_features)
    y = np.zeros(NUM_INSTANCES, dtype=np.int32) if name == "PerfectModel" else None
    return X, y


def _label(y, i):
    return None if y is None else int(y[i])


@pytest.mark.parametrize(
    "name,factory,num_features", MODEL_SPECS, ids=[spec[0] for spec in MODEL_SPECS]
)
def test_partial_methods_return_python_float(name, factory, num_features):
    X, y = _data(name, num_features)
    model = factory()

    for i in range(NUM_INSTANCES - 1):
        score = model.fit_score_partial(X[i], _label(y, i))
        assert type(score) is float, f"fit_score_partial returned {type(score)}"

    model.fit_partial(X[-1], _label(y, NUM_INSTANCES - 1))
    score = model.score_partial(X[-1])
    assert type(score) is float, f"score_partial returned {type(score)}"


@pytest.mark.parametrize(
    "name,factory,num_features", MODEL_SPECS, ids=[spec[0] for spec in MODEL_SPECS]
)
def test_batch_methods_return_1d_arrays(name, factory, num_features):
    X, y = _data(name, num_features)
    model = factory()

    scores = model.fit_score(X, y)
    assert isinstance(scores, np.ndarray)
    assert scores.dtype == np.float64
    assert scores.shape == (NUM_INSTANCES,)

    if name == "PerfectModel":  # scores only the labels it has been fitted with
        model.fit(X[:10], y[:10])

    scores = model.score(X[:10])
    assert isinstance(scores, np.ndarray)
    assert scores.dtype == np.float64
    assert scores.shape == (10,)


@pytest.mark.parametrize(
    "name,factory,num_features", MODEL_SPECS, ids=[spec[0] for spec in MODEL_SPECS]
)
def test_score_partial_before_the_first_fit_returns_a_score(name, factory, num_features):
    """Regression test for #266: fit_score_partial scores each instance before it fits it, so
    every detector must score the first instance without an error."""
    if name == "PerfectModel":  # returns the labels that it was fitted with
        pytest.skip("PerfectModel needs a label before it can score")
    X, _ = _data(name, num_features)

    score = factory().score_partial(X[0])

    assert type(score) is float
    assert np.isfinite(score)


def test_one_element_array_scores_are_converted():
    from pysad.core.base_model import BaseModel

    class ArrayModel(BaseModel):
        def fit_partial(self, X, y=None):
            return self

        def score_partial(self, X):
            return np.array([X.sum()])

    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    model = ArrayModel()

    assert type(model.score_partial(X[1])) is float
    assert model.score_partial(X[1]) == 5.0
    assert type(model.fit_score_partial(X[2])) is float
    np.testing.assert_array_equal(model.fit_score(X), [1.0, 5.0, 9.0])
    np.testing.assert_array_equal(model.score(X), [1.0, 5.0, 9.0])


def test_multiple_scores_for_one_instance_raise():
    from pysad.core.base_model import BaseModel

    class BadModel(BaseModel):
        def fit_partial(self, X, y=None):
            return self

        def score_partial(self, X):
            return np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        BadModel().score_partial(np.zeros(2))


def test_float_scores_work_downstream():
    from pysad.evaluation import AUPRMetric, AUROCMetric, WindowedMetric
    from pysad.models import xStream
    from pysad.transform.ensemble import (
        AverageScoreEnsembler,
        MaximumScoreEnsembler,
        MedianScoreEnsembler,
    )
    from pysad.transform.postprocessing import (
        AveragePostprocessor,
        ZScorePostprocessor,
    )
    from pysad.transform.probability_calibration import (
        AnomalyLikelihoodProbabilityCalibrator,
        ConformalProbabilityCalibrator,
        GaussianTailProbabilityCalibrator,
        SPOTProbabilityCalibrator,
    )

    fix_seed(61)
    X = np.random.rand(100, NUM_FEATURES)
    y = np.zeros(100, dtype=np.int32)
    y[::10] = 1

    models = [xStream(), xStream()]
    postprocessors = [
        AnomalyLikelihoodProbabilityCalibrator(learning_period=20, num_estimation_samples=20),
        ConformalProbabilityCalibrator(window_size=50),
        GaussianTailProbabilityCalibrator(window_size=50),
        SPOTProbabilityCalibrator(num_init=50, level=0.9),
        AveragePostprocessor(window_size=10),
        ZScorePostprocessor(window_size=10),
        ZScorePostprocessor(),
    ]
    ensemblers = [AverageScoreEnsembler(), MaximumScoreEnsembler(), MedianScoreEnsembler()]
    metrics = [AUROCMetric(), AUPRMetric(), WindowedMetric(AUROCMetric, window_size=50)]

    for xi, yi in zip(X, y, strict=True):
        scores = [model.fit_score_partial(xi) for model in models]

        for postprocessor in postprocessors:
            processed = postprocessor.fit_transform_partial(scores[0])
            assert np.ndim(processed) == 0

        for ensembler in ensemblers:
            combined = ensembler.fit_transform_partial(np.array(scores))
            assert np.size(combined) == 1

        for metric in metrics:
            metric.update(yi, scores[0])

    for metric in metrics:
        assert np.isfinite(metric.get())
