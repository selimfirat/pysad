def test_unsupervised_models():
    import numpy as np

    from pysad.models import (
        CPOD,
        DAMP,
        DILOF,
        KNNCAD,
        LODA,
        MCOD,
        NETS,
        SAND,
        TEDA,
        ADeMS,
        ChangeFinder,
        ExactStorm,
        HalfSpaceTrees,
        IForestASD,
        IncrementalLOF,
        Inqmad,
        KitNet,
        LocalOutlierProbability,
        MatrixProfile,
        MedianAbsoluteDeviation,
        MStream,
        NullModel,
        OnlineIsolationForest,
        OnlineOneClassSVM,
        RandomModel,
        RelativeEntropy,
        RobustRandomCutForest,
        RSForest,
        RSHash,
        SmartSifter,
        SpectralResidual,
        StandardAbsoluteDeviation,
        xStream,
    )
    from pysad.utils import fix_seed

    fix_seed(61)

    X = np.random.rand(150, 1)

    model_classes = {
        RelativeEntropy: {"min_val": 0.0, "max_val": 1.0},
        ADeMS: {},
        ChangeFinder: [{}, {"change_point": False}],
        CPOD: {},
        DILOF: {"num_neighbors": 5, "window_size": 40},
        ExactStorm: {},
        HalfSpaceTrees: {"feature_mins": [0.0], "feature_maxes": [1.0]},
        IForestASD: {},
        IncrementalLOF: {"window_size": 20},
        KitNet: {},
        KNNCAD: {"probationary_period": 50},
        LODA: {},
        LocalOutlierProbability: {"initial_X": True},
        MatrixProfile: {"window_length": 10},
        DAMP: {"window_length": 10},
        MCOD: {},
        MedianAbsoluteDeviation: [{}, {"absolute": False}],
        MStream: {},
        NETS: {},
        NullModel: {},
        OnlineIsolationForest: {},
        OnlineOneClassSVM: {},
        RandomModel: {},
        RSForest: {"window_size": 50},
        RSHash: {"feature_mins": [0.0], "feature_maxes": [1.0]},
        SAND: {"subsequence_length": 4, "batch_size": 40},
        SmartSifter: {},
        SpectralResidual: {},
        StandardAbsoluteDeviation: [{}, {"absolute": False}],
        TEDA: {},
        xStream: {},
        RobustRandomCutForest: {},
        Inqmad: {"input_shape": X.shape[1], "dim_x": 128},
    }

    for model_cls, params_dict in model_classes.items():
        if type(params_dict) is dict:
            helper_test_model(X, model_cls, params_dict)
        elif type(params_dict) is list:
            for params in params_dict:
                helper_test_model(X, model_cls, params)


def helper_test_model(X, model_cls, params_dict):
    if "initial_X" in params_dict and params_dict["initial_X"]:
        params_dict["initial_X"] = X[:25, :]
        model = model_cls(params_dict["initial_X"])
        train_X = X[25:, :]
    else:
        train_X = X
        model = model_cls(**params_dict)

    y_pred = model.fit_score(train_X)
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
    """KitNET's sigmoid saturates to 0 for large negative inputs without an overflow warning."""
    import numpy as np

    from pysad.models.kitnet_model.utils import sigmoid

    with np.errstate(over="raise"):
        y = sigmoid(np.array([-1000.0, 0.0, 1000.0]))

    np.testing.assert_array_equal(y, [0.0, 0.5, 1.0])


def test_loop_scores_without_divide_warning_on_constant_reference():
    """LoOP with identical initial instances scores new instances without a divide-by-zero warning."""
    import numpy as np

    from pysad.models import LocalOutlierProbability

    model = LocalOutlierProbability(np.zeros((20, 2)))

    with np.errstate(divide="raise"):
        assert model.score_partial(np.zeros(2)) == 0.0
        assert model.score_partial(np.ones(2)) == 1.0
