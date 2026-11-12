def test_unsupervised_models():
    import numpy as np

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
        Inqmad,
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
        RandomModel,
        RelativeEntropy,
        RobustRandomCutForest,
        RSForest,
        RSHash,
        SDOstream,
        SmartSifter,
        SpectralResidual,
        StandardAbsoluteDeviation,
        ThreshLEAP,
        xStream,
    )
    from pysad.utils import fix_seed

    fix_seed(61)

    X = np.random.rand(150, 1)

    model_classes = {
        RelativeEntropy: {"min_val": 0.0, "max_val": 1.0},
        ADeMS: {},
        AnyOut: [{}, {"outlier_score": "density"}],
        ApproxStorm: {},
        ChangeFinder: [{}, {"change_point": False}],
        CPOD: {},
        D3: {"window_size": 50},
        DILOF: {"num_neighbors": 5, "window_size": 40},
        ExactStorm: {},
        EXPoSE: {"num_random_features": 50},
        HalfSpaceTrees: {"feature_mins": [0.0], "feature_maxes": [1.0]},
        HoltWinters: [{"period": 10}, {"period": 10, "threshold": 1}],
        IForestASD: {},
        IncrementalLOF: {"window_size": 20},
        KELOS: {},
        KitNet: {},
        KNNCAD: {"probationary_period": 50},
        LODA: {},
        LocalOutlierProbability: {"initial_X": True},
        MatrixProfile: {"window_length": 10},
        DAMP: {"window_length": 10},
        MCOD: {},
        MedianAbsoluteDeviation: [{}, {"absolute": False}],
        MemStream: {"window_size": 8, "num_epochs": 20},
        MStream: {},
        NETS: {},
        NullModel: {},
        OeSNNUAD: [{}, {"window_size": 10, "num_output_neurons": 5}],
        OnlineIsolationForest: {},
        OnlineOneClassSVM: {},
        PCBIsolationForest: {
            "num_trees": 10,
            "window_size": 40,
            "stat_size": 10,
        },
        PEWMA: [{}, {"beta": 0.0}],
        RandomModel: {},
        RSForest: {"window_size": 50},
        RSHash: {"feature_mins": [0.0], "feature_maxes": [1.0]},
        SAND: {"subsequence_length": 4, "batch_size": 40},
        SDOstream: {},
        SmartSifter: {},
        SpectralResidual: {},
        StandardAbsoluteDeviation: [{}, {"absolute": False}],
        TEDA: {},
        ThreshLEAP: {},
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

    # HTM takes each instance as [timestamp, value].
    X_timestamped = np.column_stack([3600.0 * np.arange(len(X)), X])
    for likelihood in (True, False):
        params = {
            "min_val": 0.0,
            "max_val": 1.0,
            "probationary_period": 50,
            "likelihood": likelihood,
        }
        helper_test_model(X_timestamped, HTM, params)

    # MIDAS takes each instance as an edge [source, destination].
    X_edges = np.random.RandomState(0).randint(0, 10, size=(len(X), 2)).astype(np.float64)
    helper_test_model(X_edges, MIDAS, {"tick_size": 10})


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
