import numpy as np
import pytest

from pysad.utils import fix_seed

NUM_INSTANCES = 150
NUM_FEATURES = 3


def _initial_X(num_features):
    return np.random.RandomState(0).rand(25, num_features)


def _model_specs():
    from pysad import models
    from pysad.models import (ExactStorm, HalfSpaceTrees, IForestASD, KitNet, KNNCAD, LODA,
                              LocalOutlierProbability, MedianAbsoluteDeviation, NullModel,
                              PerfectModel, RandomModel, RelativeEntropy, RobustRandomCutForest,
                              RSHash, SeasonalESD, SeasonalHybridESD, StandardAbsoluteDeviation,
                              xStream)
    from pysad.models.integrations import OneFitModel, ReferenceWindowModel
    from pyod.models.iforest import IForest

    mins, maxes = [0.0] * NUM_FEATURES, [1.0] * NUM_FEATURES

    # (name, factory, number of features)
    specs = [
        ("ExactStorm", lambda: ExactStorm(window_size=20), NUM_FEATURES),
        ("HalfSpaceTrees", lambda: HalfSpaceTrees(feature_mins=mins, feature_maxes=maxes, window_size=20, num_trees=5, max_depth=5), NUM_FEATURES),
        ("IForestASD", lambda: IForestASD(window_size=32), NUM_FEATURES),
        ("KitNet", lambda: KitNet(grace_feature_mapping=10, grace_anomaly_detector=10), NUM_FEATURES),
        ("KNNCAD", lambda: KNNCAD(probationary_period=50), 1),
        ("LODA", lambda: LODA(), NUM_FEATURES),
        ("LocalOutlierProbability", lambda: LocalOutlierProbability(_initial_X(NUM_FEATURES)), NUM_FEATURES),
        ("MedianAbsoluteDeviation", lambda: MedianAbsoluteDeviation(), 1),
        ("MedianAbsoluteDeviation-signed", lambda: MedianAbsoluteDeviation(absolute=False), 1),
        ("NullModel", lambda: NullModel(), NUM_FEATURES),
        ("PerfectModel", lambda: PerfectModel(), NUM_FEATURES),
        ("RandomModel", lambda: RandomModel(), NUM_FEATURES),
        ("RelativeEntropy", lambda: RelativeEntropy(min_val=0.0, max_val=1.0), 1),
        ("RobustRandomCutForest", lambda: RobustRandomCutForest(num_trees=4, tree_size=32), NUM_FEATURES),
        ("RSHash", lambda: RSHash(feature_mins=mins, feature_maxes=maxes, sampling_points=50), NUM_FEATURES),
        ("SeasonalESD", lambda: SeasonalESD(period=4, window_size=12, max_anomalies=2), 1),
        ("SeasonalHybridESD", lambda: SeasonalHybridESD(period=4, window_size=12, max_anomalies=2), 1),
        ("StandardAbsoluteDeviation", lambda: StandardAbsoluteDeviation(), 1),
        ("StandardAbsoluteDeviation-signed", lambda: StandardAbsoluteDeviation(absolute=False), 1),
        ("xStream", lambda: xStream(), NUM_FEATURES),
        ("OneFitModel", lambda: OneFitModel(IForest, _initial_X(NUM_FEATURES)), NUM_FEATURES),
        ("ReferenceWindowModel", lambda: ReferenceWindowModel(IForest, window_size=20, sliding_size=10, initial_window_X=_initial_X(NUM_FEATURES)), NUM_FEATURES),
    ]

    if getattr(models, "_has_inqmad", False):
        specs.append(("Inqmad", lambda: models.Inqmad(input_shape=NUM_FEATURES, dim_x=32, gamma=10), NUM_FEATURES))

    return specs


MODEL_SPECS = _model_specs()


def _data(name, num_features):
    fix_seed(61)
    X = np.random.rand(NUM_INSTANCES, num_features)
    y = np.zeros(NUM_INSTANCES, dtype=np.int32) if name == "PerfectModel" else None
    return X, y


def _label(y, i):
    return None if y is None else int(y[i])


@pytest.mark.parametrize("name,factory,num_features", MODEL_SPECS, ids=[spec[0] for spec in MODEL_SPECS])
def test_partial_methods_return_python_float(name, factory, num_features):
    X, y = _data(name, num_features)
    model = factory()

    for i in range(NUM_INSTANCES - 1):
        score = model.fit_score_partial(X[i], _label(y, i))
        assert type(score) is float, "fit_score_partial returned {}".format(type(score))

    model.fit_partial(X[-1], _label(y, NUM_INSTANCES - 1))
    score = model.score_partial(X[-1])
    assert type(score) is float, "score_partial returned {}".format(type(score))


@pytest.mark.parametrize("name,factory,num_features", MODEL_SPECS, ids=[spec[0] for spec in MODEL_SPECS])
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
    from pysad.transform.ensemble import AverageScoreEnsembler, MaximumScoreEnsembler, MedianScoreEnsembler
    from pysad.transform.postprocessing import RunningAveragePostprocessor, RunningZScorePostprocessor, ZScorePostprocessor
    from pysad.transform.probability_calibration import ConformalProbabilityCalibrator, GaussianTailProbabilityCalibrator

    fix_seed(61)
    X = np.random.rand(100, NUM_FEATURES)
    y = np.zeros(100, dtype=np.int32)
    y[::10] = 1

    models = [xStream(), xStream()]
    postprocessors = [
        ConformalProbabilityCalibrator(window_size=50),
        GaussianTailProbabilityCalibrator(window_size=50),
        RunningAveragePostprocessor(window_size=10),
        RunningZScorePostprocessor(window_size=10),
        ZScorePostprocessor(),
    ]
    ensemblers = [AverageScoreEnsembler(), MaximumScoreEnsembler(), MedianScoreEnsembler()]
    metrics = [AUROCMetric(), AUPRMetric(), WindowedMetric(AUROCMetric, window_size=50)]

    for xi, yi in zip(X, y):
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
