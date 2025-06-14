def test_reference_window(test_path):
    from sklearn.utils import shuffle
    from pysad.models.integrations import ReferenceWindowModel
    from pysad.utils import Data
    from pysad.evaluation import AUROCMetric
    from pysad.utils import ArrayStreamer
    import os
    from pyod.models.iforest import IForest

    data = Data(os.path.join(test_path,"../../examples/data"))

    X_all, y_all = data.get_data("arrhythmia.mat")
    X_all, y_all = shuffle(X_all, y_all)

    model = ReferenceWindowModel(model_cls=IForest, window_size=240, sliding_size=30,
                                 initial_window_X=X_all[:100])

    iterator = ArrayStreamer(shuffle=False)

    auroc = AUROCMetric()

    y_pred = []
    for X, y in iterator.iter(X_all[100:], y_all[100:]):
        model.fit_partial(X)
        score = model.score_partial(X)

        y_pred.append(score)

        auroc.update(y, score)

    print("AUROC: ", auroc.get())


def test_one_fit(test_path):
    from sklearn.utils import shuffle
    from pysad.utils import Data
    from pysad.evaluation import AUROCMetric
    from pysad.utils import ArrayStreamer
    import os
    from pyod.models.iforest import IForest
    from pysad.models.integrations.one_fit_model import OneFitModel

    data = Data(os.path.join(test_path, "../../examples/data"))

    X_all, y_all = data.get_data("arrhythmia.mat")
    print(X_all, y_all)
    X_all, y_all = shuffle(X_all, y_all)

    model = OneFitModel(model_cls=IForest, initial_X=X_all[:100])

    iterator = ArrayStreamer(shuffle=False)

    auroc = AUROCMetric()

    y_pred = []
    for X, y in iterator.iter(X_all[100:], y_all[100:]):
        model.fit_partial(X)
        score = model.score_partial(X)

        y_pred.append(score)

        auroc.update(y, score)

    print("AUROC: ", auroc.get())


def test_one_fit_model_basic():
    """Test OneFitModel basic functionality."""
    from pysad.models.integrations.one_fit_model import OneFitModel
    from pyod.models.iforest import IForest
    import numpy as np
    
    # Create test data
    np.random.seed(42)
    initial_X = np.random.random((50, 2))
    test_X = np.random.random((20, 2))
    
    # Test OneFitModel
    model = OneFitModel(model_cls=IForest, initial_X=initial_X)
    
    # Test that fit_partial does nothing but returns self
    result = model.fit_partial(test_X[0])
    assert result is model
    
    # Test scoring
    for i in range(len(test_X)):
        score = model.score_partial(test_X[i])
        assert isinstance(score, (int, float))


def test_one_fit_model_with_labels():
    """Test OneFitModel with supervised learning."""
    from pysad.models.integrations.one_fit_model import OneFitModel
    from pyod.models.iforest import IForest
    import numpy as np
    import warnings
    
    # Create test data with labels
    np.random.seed(42)
    initial_X = np.random.random((50, 2))
    
    # Test OneFitModel with labels
    model = OneFitModel(model_cls=IForest, initial_X=initial_X)
    
    # Test scoring
    test_instance = np.random.random(2)
    score = model.score_partial(test_instance)
    assert isinstance(score, (int, float))


def test_reference_window_model_basic():
    """Test ReferenceWindowModel basic functionality."""
    from pysad.models.integrations.reference_window_model import ReferenceWindowModel
    from pyod.models.iforest import IForest
    import numpy as np
    
    # Create test data
    np.random.seed(42)
    initial_X = np.random.random((30, 2))
    
    # Test ReferenceWindowModel
    model = ReferenceWindowModel(
        model_cls=IForest,
        window_size=20,
        sliding_size=5,
        initial_window_X=initial_X
    )
    
    # Test fitting and scoring
    test_instances = np.random.random((15, 2))
    
    for i, instance in enumerate(test_instances):
        model.fit_partial(instance)
        score = model.score_partial(instance)
        assert isinstance(score, (int, float))


def test_reference_window_model_without_initial():
    """Test ReferenceWindowModel without initial window."""
    from pysad.models.integrations.reference_window_model import ReferenceWindowModel
    from pyod.models.iforest import IForest
    import numpy as np
    
    # Test ReferenceWindowModel without initial window
    model = ReferenceWindowModel(
        model_cls=IForest,
        window_size=10,
        sliding_size=3
    )
    
    # Test fitting and scoring
    np.random.seed(42)
    test_instances = np.random.random((25, 2))
    
    for i, instance in enumerate(test_instances):
        model.fit_partial(instance)
        score = model.score_partial(instance)
        assert isinstance(score, (int, float))


def test_reference_window_model_with_labels():
    """Test ReferenceWindowModel with labels."""
    from pysad.models.integrations.reference_window_model import ReferenceWindowModel
    from pyod.models.iforest import IForest
    import numpy as np
    import warnings
    
    # Create test data with labels
    np.random.seed(42)
    initial_X = np.random.random((20, 2))
    initial_y = np.random.randint(0, 2, 20)
    
    # Test ReferenceWindowModel with labels
    # Suppress PyOD warning about labels in unsupervised learning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y should not be presented in unsupervised learning")
        model = ReferenceWindowModel(
            model_cls=IForest,
            window_size=15,
            sliding_size=5,
            initial_window_X=initial_X,
            initial_window_y=initial_y
        )
    
        # Test fitting and scoring
        test_instances = np.random.random((10, 2))
        test_labels = np.random.randint(0, 2, 10)
        
        for i, (instance, label) in enumerate(zip(test_instances, test_labels)):
            model.fit_partial(instance, label)
            score = model.score_partial(instance)
            assert isinstance(score, (int, float))


def test_reference_window_model_window_update():
    """Test ReferenceWindowModel window update mechanism."""
    from pysad.models.integrations.reference_window_model import ReferenceWindowModel
    from pyod.models.iforest import IForest
    import numpy as np
    
    # Test with small window and sliding size for easier testing
    np.random.seed(42)
    initial_X = np.random.random((5, 2))
    
    model = ReferenceWindowModel(
        model_cls=IForest,
        window_size=5,
        sliding_size=2,  # Model should retrain every 2 instances
        initial_window_X=initial_X
    )
    
    # Add instances and verify model retrains
    test_instances = np.random.random((8, 2))
    
    for i, instance in enumerate(test_instances):
        model.fit_partial(instance)
        score = model.score_partial(instance)
        assert isinstance(score, (int, float))
        
        # Check internal state
        assert len(model.cur_window_X) <= model.window_size
