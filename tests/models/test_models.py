import copy

import numpy as np
import pytest

import pysad.models
from pysad.transform.postprocessing import ZScorePostprocessor
from pysad.transform.preprocessing import ShingleTransformer
from pysad.utils import fix_seed
from tests.models.conftest import EDGE_MODELS, TIMESTAMPED_MODELS, active_model_names

MODEL_CASES = {
    "DriftAdaptiveDetector": [{"model_factory": pysad.models.LODA}],
    "AnyOut": [{}],
    "ChangeFinder": [{}],
    "D3": [{"window_size": 30}],
    "DAMP": [{"window_length": 10}],
    "DILOF": [{"num_neighbors": 5, "window_size": 24}],
    "EXPoSE": [{"num_random_features": 50}],
    "FeedbackHST": [
        {
            "feature_mins": [0.0],
            "feature_maxes": [1.0],
            "window_size": 10,
            "num_trees": 5,
            "max_depth": 5,
            "feedback_budget": 10,
        }
    ],
    "HalfSpaceTrees": [
        {"feature_mins": [0.0], "feature_maxes": [1.0], "num_trees": 5, "max_depth": 5}
    ],
    "HoltWinters": [{"period": 10}],
    "HTM": [
        {
            "min_val": 0.0,
            "max_val": 1.0,
            "probationary_period": 30,
            "num_columns": 64,
            "likelihood": True,
        },
        {
            "min_val": 0.0,
            "max_val": 1.0,
            "probationary_period": 30,
            "num_columns": 64,
            "likelihood": False,
        },
    ],
    "IForestASD": [{"window_size": 20, "n_estimators": 5}],
    "IncrementalLOF": [{"window_size": 20}],
    "IDKS": [{"input_shape": 1, "window_size": 32, "num_samples": 4, "num_partitions": 10}],
    "Inqmad": [{"input_shape": 1, "dim_x": 32}],
    "KNNCAD": [{"probationary_period": 48}],
    "LocalOutlierProbability": [{"initial_X": True}],
    "MatrixProfile": [{"window_length": 10}],
    "MedianAbsoluteDeviation": [{}],
    "MemStream": [{"window_size": 8, "num_epochs": 20}],
    "MIDAS": [{"tick_size": 10}],
    "MiLOF": [{"window_size": 30, "num_summaries": 5}],
    "OeSNNUAD": [{"window_size": 10, "num_output_neurons": 5}],
    "OnlineIsolationForest": [{"num_trees": 5, "window_size": 24}],
    "PCBIsolationForest": [{"num_trees": 10, "window_size": 30, "stat_size": 10}],
    "PEWMA": [{}],
    "RelativeEntropy": [{"min_val": 0.0, "max_val": 1.0}],
    "RobustRandomCutForest": [{"num_trees": 4, "tree_size": 32}],
    "RSForest": [{"window_size": 30, "num_trees": 5, "max_depth": 5}],
    "RSHash": [{"feature_mins": [0.0], "feature_maxes": [1.0], "num_components": 10}],
    "SAND": [{"subsequence_length": 4, "batch_size": 30}],
    "SeasonalESD": [{"period": 10, "window_size": 31}],
    "SeasonalHybridESD": [{"period": 10, "window_size": 31}],
    "StandardAbsoluteDeviation": [{}],
    "ZeroShotEmbeddingDetector": [{"semantic_anchors": np.random.RandomState(0).rand(10, 1)}],
    "ModelEnsemble": [
        {
            "models": [
                pysad.models.LODA(num_random_cuts=10, random_state=0)
                | ZScorePostprocessor(window_size=20),
                pysad.models.xStream(n_chains=10, random_state=1)
                | ZScorePostprocessor(window_size=20),
            ]
        }
    ],
    "StreamPipeline": [
        {
            "model": pysad.models.LODA(num_random_cuts=10),
            "preprocessors": [ShingleTransformer(shingle_size=2)],
            "postprocessors": [ZScorePostprocessor(window_size=20)],
        }
    ],
}


def _parametrized_cases():
    cases = []
    for name in active_model_names():
        param_list = MODEL_CASES.get(name, [{}])
        for index, params in enumerate(param_list):
            case_id = name if len(param_list) == 1 else f"{name}-{index}"
            cases.append(pytest.param(name, params, id=case_id))
    return cases


@pytest.mark.parametrize("model_name,params_dict", _parametrized_cases())
def test_unsupervised_models(model_name, params_dict):
    fix_seed(61)
    X = np.random.rand(80, 1)
    if model_name in TIMESTAMPED_MODELS:
        X = np.column_stack([3600.0 * np.arange(len(X)), X])
    elif model_name in EDGE_MODELS:
        X = np.random.RandomState(0).randint(0, 10, size=(len(X), 2)).astype(np.float64)

    helper_test_model(X, getattr(pysad.models, model_name), copy.deepcopy(params_dict))


def helper_test_model(X, model_cls, params_dict):
    if "initial_X" in params_dict and params_dict["initial_X"]:
        params_dict["initial_X"] = X[:25, :]
        model = model_cls(params_dict["initial_X"])
        train_X = X[25:, :]
    else:
        train_X = X
        model = model_cls(**params_dict)

    y = np.zeros(train_X.shape[0], dtype=np.int32) if model_cls.__name__ == "PerfectModel" else None
    y_pred = model.fit_score(train_X, y)
    assert y_pred.shape == (train_X.shape[0],)


def test_relative_entropy_no_warnings():
    """Test that RelativeEntropy doesn't produce DeprecationWarnings about scalar conversion."""
    import warnings

    import numpy as np

    from pysad.models import RelativeEntropy
    from pysad.utils import fix_seed

    fix_seed(61)
    X = np.random.rand(150, 1)

    # Capture any DeprecationWarnings about scalar conversion
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        model = RelativeEntropy(min_val=0.0, max_val=1.0)
        y_pred = model.fit_score(X)

        # Filter for the specific warnings we're concerned about
        scalar_warnings = [
            warning
            for warning in w
            if issubclass(warning.category, DeprecationWarning)
            and (
                "scalar divide" in str(warning.message)
                or "Conversion of an array with ndim > 0 to a scalar" in str(warning.message)
            )
        ]

        # Assert no scalar conversion warnings occurred
        assert len(scalar_warnings) == 0, f"Found {len(scalar_warnings)} scalar conversion warnings"
        assert y_pred.shape == (X.shape[0],)


def test_kitnet_sigmoid_does_not_overflow():
    """KitNET scores extreme negative and positive inputs without a sigmoid overflow warning."""
    import numpy as np

    from pysad.models import KitNet

    model = KitNet(grace_feature_mapping=2, grace_anomaly_detector=2, random_state=0)
    model.fit(np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.8]]))

    with np.errstate(over="raise"):
        score = model.score_partial(np.array([-1e4, 1e4]))

    assert np.isfinite(score)
    assert score >= 0.0


def test_loop_scores_without_divide_warning_on_constant_reference():
    """LoOP with identical initial instances scores new instances without a divide-by-zero warning."""
    import numpy as np

    from pysad.models import LocalOutlierProbability

    model = LocalOutlierProbability(np.zeros((20, 2)))

    with np.errstate(divide="raise"):
        assert model.score_partial(np.zeros(2)) == 0.0
        assert model.score_partial(np.ones(2)) == 1.0
