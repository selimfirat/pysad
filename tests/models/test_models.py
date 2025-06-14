def test_unsupervised_models():
    from pysad.models import RobustRandomCutForest
    from pysad.models import ExactStorm
    from pysad.models import HalfSpaceTrees
    from pysad.models import IForestASD
    from pysad.models import KitNet
    from pysad.models import KNNCAD
    from pysad.models import LODA
    from pysad.models import LocalOutlierProbability
    from pysad.models import MedianAbsoluteDeviation
    from pysad.models import NullModel
    from pysad.models import RandomModel
    from pysad.models import RelativeEntropy
    from pysad.models import RSHash
    from pysad.models import StandardAbsoluteDeviation
    from pysad.models import xStream
    from pysad.models import Inqmad
    import numpy as np
    from pysad.utils import fix_seed
    fix_seed(61)

    X = np.random.rand(150, 1)

    model_classes = {
        RelativeEntropy: {"min_val": 0.0, "max_val": 1.0},
        ExactStorm: {},
        HalfSpaceTrees: {"feature_mins": [0.0], "feature_maxes": [1.0]},
        IForestASD: {},
        KitNet: {},
        KNNCAD: {"probationary_period": 50},
        LODA: {},
        LocalOutlierProbability: { "initial_X": True },
        MedianAbsoluteDeviation: [{}, { "absolute": False }],
        NullModel: {},
        RandomModel: {},
        RSHash: {"feature_mins": [0.0], "feature_maxes": [1.0]},
        StandardAbsoluteDeviation: [{}, {"absolute": False}],
        xStream: {},
        RobustRandomCutForest: {},
        Inqmad: {"input_shape": X.shape[1], "dim_x": 128, "gamma": 100}
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


def test_fit_and_score_separately():
    from pysad.models import xStream
    import numpy as np
    from pysad.utils import fix_seed
    fix_seed(61)

    X = np.random.rand(150, 1)

    model = xStream()

    model = model.fit(X)
    y_pred = model.score(X)
    assert y_pred.shape == (X.shape[0],)


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
        scalar_warnings = [warning for warning in w 
                          if issubclass(warning.category, DeprecationWarning) and
                          ('scalar divide' in str(warning.message) or 
                           'Conversion of an array with ndim > 0 to a scalar' in str(warning.message))]
        
        # Assert no scalar conversion warnings occurred
        assert len(scalar_warnings) == 0, f"Found {len(scalar_warnings)} scalar conversion warnings"
        assert y_pred.shape == (X.shape[0],)
