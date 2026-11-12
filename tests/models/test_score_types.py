import importlib.util

import numpy as np
import pytest

from pysad.utils import fix_seed

NUM_INSTANCES = 150
NUM_FEATURES = 3


def _initial_X(num_features):
    return np.random.RandomState(0).rand(25, num_features)


def _model_specs():
    from pyod.models.iforest import IForest

    from pysad import models
    from pysad.models import (
        CPOD,
        D3,
        DAMP,
        DILOF,
        HTM,
        KELOS,
        KNNCAD,
        LODA,
        MCOD,
        MIDAS,
        NETS,
        PEWMA,
        SAND,
        TEDA,
        ADeMS,
        AnyOut,
        ApproxStorm,
        ChangeFinder,
        ExactStorm,
        EXPoSE,
        HalfSpaceTrees,
        HoltWinters,
        IForestASD,
        IncrementalLOF,
        KitNet,
        LocalOutlierProbability,
        MatrixProfile,
        MedianAbsoluteDeviation,
        MemStream,
        MStream,
        NullModel,
        OeSNNUAD,
        OnlineIsolationForest,
        OnlineOneClassSVM,
        PCBIsolationForest,
        PerfectModel,
        RandomModel,
        RelativeEntropy,
        RobustRandomCutForest,
        RSForest,
        RSHash,
        SDOstream,
        SeasonalESD,
        SeasonalHybridESD,
        SmartSifter,
        SpectralResidual,
        StandardAbsoluteDeviation,
        StreamPipeline,
        ThreshLEAP,
        xStream,
    )
    from pysad.models.integrations import OneFitModel, ReferenceWindowModel
    from pysad.transform.postprocessing import RunningZScorePostprocessor
    from pysad.transform.preprocessing import ShingleTransformer

    mins, maxes = [0.0] * NUM_FEATURES, [1.0] * NUM_FEATURES

    # (name, factory, number of features)
    specs = [
        ("ADeMS", lambda: ADeMS(), NUM_FEATURES),
        ("AnyOut", lambda: AnyOut(max_height=2, outlier_score="density"), NUM_FEATURES),
        (
            "ApproxStorm",
            lambda: ApproxStorm(
                window_size=20, max_radius=0.3, num_neighbors=3, sample_fraction=0.25
            ),
            NUM_FEATURES,
        ),
        ("ChangeFinder", lambda: ChangeFinder(), 1),
        ("ChangeFinder-outlier", lambda: ChangeFinder(change_point=False), 1),
        ("CPOD", lambda: CPOD(window_size=20, max_radius=0.3), NUM_FEATURES),
        ("D3", lambda: D3(window_size=20, sample_fraction=0.5, max_radius=0.3), NUM_FEATURES),
        ("DILOF", lambda: DILOF(num_neighbors=5, window_size=40), NUM_FEATURES),
        ("ExactStorm", lambda: ExactStorm(window_size=20), NUM_FEATURES),
        ("EXPoSE", lambda: EXPoSE(num_random_features=50), NUM_FEATURES),
        (
            "HalfSpaceTrees",
            lambda: HalfSpaceTrees(
                feature_mins=mins, feature_maxes=maxes, window_size=20, num_trees=5, max_depth=5
            ),
            NUM_FEATURES,
        ),
        ("HoltWinters", lambda: HoltWinters(period=4, threshold=1), 1),
        # HTM takes each instance as [timestamp, value].
        ("HTM", lambda: HTM(min_val=0.0, max_val=1.0, probationary_period=20, num_columns=256), 2),
        (
            "HTM-raw",
            lambda: HTM(min_val=0.0, max_val=1.0, likelihood=False, num_columns=256),
            2,
        ),
        ("IForestASD", lambda: IForestASD(window_size=32), NUM_FEATURES),
        ("IncrementalLOF", lambda: IncrementalLOF(window_size=20), NUM_FEATURES),
        ("KELOS", lambda: KELOS(window_size=20, radius=0.3, num_neighbors=3), NUM_FEATURES),
        (
            "KitNet",
            lambda: KitNet(grace_feature_mapping=10, grace_anomaly_detector=10),
            NUM_FEATURES,
        ),
        ("KNNCAD", lambda: KNNCAD(probationary_period=50), 1),
        ("LODA", lambda: LODA(), NUM_FEATURES),
        (
            "LocalOutlierProbability",
            lambda: LocalOutlierProbability(_initial_X(NUM_FEATURES)),
            NUM_FEATURES,
        ),
        ("MatrixProfile", lambda: MatrixProfile(window_length=8, history_size=40), 1),
        ("DAMP", lambda: DAMP(window_length=8, history_size=40), 1),
        ("MCOD", lambda: MCOD(window_size=20, max_radius=0.3, num_neighbors=2), NUM_FEATURES),
        ("MedianAbsoluteDeviation", lambda: MedianAbsoluteDeviation(), 1),
        ("MedianAbsoluteDeviation-signed", lambda: MedianAbsoluteDeviation(absolute=False), 1),
        # MIDAS takes each instance as an edge [source, destination].
        ("MIDAS", lambda: MIDAS(tick_size=5), 2),
        ("MemStream", lambda: MemStream(window_size=8, num_epochs=20), NUM_FEATURES),
        ("MStream", lambda: MStream(), NUM_FEATURES),
        ("NETS", lambda: NETS(window_size=20, max_radius=0.3), NUM_FEATURES),
        ("NullModel", lambda: NullModel(), NUM_FEATURES),
        ("OeSNNUAD", lambda: OeSNNUAD(window_size=20), 1),
        ("OnlineIsolationForest", lambda: OnlineIsolationForest(), NUM_FEATURES),
        ("OnlineOneClassSVM", lambda: OnlineOneClassSVM(), NUM_FEATURES),
        (
            "PCBIsolationForest",
            lambda: PCBIsolationForest(num_trees=10, window_size=32, stat_size=10),
            NUM_FEATURES,
        ),
        ("PerfectModel", lambda: PerfectModel(), NUM_FEATURES),
        ("PEWMA", lambda: PEWMA(), 1),
        ("RandomModel", lambda: RandomModel(), NUM_FEATURES),
        # step=1 so that every value from the 10th on, including the last score_partial, is scored against the hypotheses
        (
            "RelativeEntropy",
            lambda: RelativeEntropy(min_val=0.0, max_val=1.0, window_size=10, step=1),
            1,
        ),
        (
            "RobustRandomCutForest",
            lambda: RobustRandomCutForest(num_trees=4, tree_size=32),
            NUM_FEATURES,
        ),
        (
            "Inqmad",
            lambda: models.Inqmad(input_shape=NUM_FEATURES, dim_x=32),
            NUM_FEATURES,
        ),
        ("RSForest", lambda: RSForest(window_size=20, num_trees=5, max_depth=8), NUM_FEATURES),
        ("RSHash", lambda: RSHash(feature_mins=mins, feature_maxes=maxes), NUM_FEATURES),
        ("SAND", lambda: SAND(subsequence_length=4, batch_size=40), 1),
        ("SDOstream", lambda: SDOstream(), NUM_FEATURES),
        ("SeasonalESD", lambda: SeasonalESD(period=4, window_size=12, max_anomalies=0.17), 1),
        (
            "SeasonalHybridESD",
            lambda: SeasonalHybridESD(period=4, window_size=12, max_anomalies=0.17),
            1,
        ),
        ("SmartSifter", lambda: SmartSifter(), NUM_FEATURES),
        ("SpectralResidual", lambda: SpectralResidual(window_size=30), 1),
        ("StandardAbsoluteDeviation", lambda: StandardAbsoluteDeviation(), 1),
        ("StandardAbsoluteDeviation-signed", lambda: StandardAbsoluteDeviation(absolute=False), 1),
        (
            "StreamPipeline",
            lambda: StreamPipeline(
                model=LODA(),
                preprocessors=[ShingleTransformer(shingle_size=2)],
                postprocessors=[RunningZScorePostprocessor(window_size=20)],
            ),
            NUM_FEATURES,
        ),
        ("TEDA", lambda: TEDA(), NUM_FEATURES),
        (
            "ThreshLEAP",
            lambda: ThreshLEAP(window_size=20, max_radius=0.3, num_neighbors=3),
            NUM_FEATURES,
        ),
        ("xStream", lambda: xStream(), NUM_FEATURES),
        ("OneFitModel", lambda: OneFitModel(IForest, _initial_X(NUM_FEATURES)), NUM_FEATURES),
        (
            "ReferenceWindowModel",
            lambda: ReferenceWindowModel(
                IForest, window_size=20, sliding_size=10, initial_window_X=_initial_X(NUM_FEATURES)
            ),
            NUM_FEATURES,
        ),
    ]

    # River needs Python 3.11 or newer, so the test group installs it only there.
    if importlib.util.find_spec("river") is not None:
        from river import time_series

        from pysad.models.integrations import PredictiveAnomalyDetector

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
        RunningAveragePostprocessor,
        RunningZScorePostprocessor,
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
        RunningAveragePostprocessor(window_size=10),
        RunningZScorePostprocessor(window_size=10),
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
