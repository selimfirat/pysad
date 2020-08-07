
def test_unsupervised_models():
    from pysad.models.exact_storm import ExactStorm
    from pysad.models.half_space_trees import HalfSpaceTrees
    from pysad.models.iforest_asd import IForestASD
    from pysad.models.kitnet import KitNet
    from pysad.models.knn_cad import KNNCAD
    from pysad.models.loda import LODA
    from pysad.models.loop import StreamLocalOutlierProbability
    from pysad.models.median_absolute_deviation import MedianAbsoluteDeviation
    from pysad.models.null_model import NullModel
    from pysad.models.random_model import RandomModel
    from pysad.models.relative_entropy import RelativeEntropy
    from pysad.models.rs_hash import RSHash
    from pysad.models.standard_absolute_deviation import StandardAbsoluteDeviation
    from pysad.models.xstream import xStream
    import numpy as np
    np.random.seed(61)

    X = np.random.rand(150, 1)

    model_classes = {
        ExactStorm: {},
        HalfSpaceTrees: {"feature_mins": [0.0], "feature_maxes": [1.0]},
        IForestASD: {},
        KitNet: {},
        KNNCAD: { "probationary_period": 50 },
        LODA: { },
        StreamLocalOutlierProbability: { "initial_X": True },
        MedianAbsoluteDeviation: {},
        NullModel: {},
        RandomModel: {},
        RelativeEntropy: { "min_val": 0.0, "max_val": 1.0 },
        RSHash: {"feature_mins": [0.0], "feature_maxes": [1.0]},
        StandardAbsoluteDeviation: {},
        xStream: {}
    }

    for model_cls, params_dict in model_classes.items():
        print(str(model_cls))
        if "initial_X" in params_dict and params_dict["initial_X"]:
            params_dict["initial_X"] = X[:25,:]
            model = model_cls(params_dict["initial_X"])
            train_X = X[25:,:]
        else:
            train_X = X
            model = model_cls(**params_dict)

        model.fit_score(train_X)
