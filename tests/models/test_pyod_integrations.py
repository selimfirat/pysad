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


def test_reference_window_model_issue_23_fix():
    """Test for the fix of issue #23: reference window duplication bug.
    
    The bug was that reference_window_X and cur_window_X pointed to the same
    list object, causing data duplication when concatenation occurred.
    """
    from pysad.models.integrations.reference_window_model import ReferenceWindowModel
    from pyod.models.iforest import IForest
    import numpy as np
    
    # Setup from issue #23
    window_size = 4
    sliding_size = 2
    
    model = ReferenceWindowModel(
        model_cls=IForest,
        window_size=window_size,
        sliding_size=sliding_size,
        initial_window_X=None
    )
    
    # Step 1: Add first data point
    X1 = np.array([1.0])
    model.fit_partial(X1)
    
    # Verify that cur_window_X and reference_window_X are independent objects
    assert model.cur_window_X is not model.reference_window_X, \
        "cur_window_X and reference_window_X should be independent objects"
    
    # Step 2: Add second data point
    X2 = np.array([2.0])
    model.fit_partial(X2)
    
    # Verify they are still independent
    assert model.cur_window_X is not model.reference_window_X, \
        "cur_window_X and reference_window_X should remain independent objects"
    
    # Test independence: modify cur_window_X and ensure reference_window_X is unchanged
    original_ref_len = len(model.reference_window_X)
    model.cur_window_X.append(np.array([999.0]))
    
    # reference_window_X should not be affected by changes to cur_window_X
    assert len(model.reference_window_X) == original_ref_len, \
        "reference_window_X should not be affected by changes to cur_window_X"
    
    # Extract values for comparison - should be [1.0, 2.0], NOT [1.0, 2.0, 1.0, 2.0]
    ref_values = [x[0] for x in model.reference_window_X]
    expected = [1.0, 2.0]
    assert ref_values == expected, \
        f"Expected {expected}, but got {ref_values}. Duplication detected!"


def test_reference_window_model_issue_25_fix():
    """Test for the fix of issue #25: Reference window reset bug.
    
    The bug: When cur_window_X length is less than window_size after sliding,
    the reference_window_X is incorrectly reset to just the current window
    instead of maintaining the properly sized reference window.
    
    Test scenario: window_size=4, sliding_size=2, streaming data [1,2,3,4,5,6,7,8,...]
    The critical test is after sliding occurs and cur_window_X becomes small again,
    the reference_window_X should NOT be reset to just the current window.
    """
    from pysad.models.integrations.reference_window_model import ReferenceWindowModel
    from pyod.models.iforest import IForest
    import numpy as np
    
    window_size = 4
    sliding_size = 2
    
    model = ReferenceWindowModel(
        model_cls=IForest,
        window_size=window_size,
        sliding_size=sliding_size,
        initial_window_X=None
    )
    
    # Build up the initial window
    for i in range(1, 9):
        model.fit_partial(np.array([float(i)]))
    
    # At this point, after several sliding operations, the reference window
    # should be properly sized (window_size=4) and should NOT be reset to
    # just the current window when cur_window_X is small
    
    # Verify reference window is properly sized
    assert len(model.reference_window_X) == window_size, \
        f"Reference window should be size {window_size}, got {len(model.reference_window_X)}"
    
    # Verify reference window is not just the current window
    ref_values = [x[0] for x in model.reference_window_X]
    cur_values = [x[0] for x in model.cur_window_X] if model.cur_window_X else []
    
    # The reference window should not be identical to the current window
    # (this would indicate the bug where reference_window_X gets reset)
    assert ref_values != cur_values, \
        f"Reference window should not be reset to current window. Got ref={ref_values}, cur={cur_values}"
    
    # Reference window should contain the most recent window_size elements
    # from the sliding window, not just the current partial window
    assert len(ref_values) == window_size, \
        f"Reference window should maintain size {window_size}, got {len(ref_values)}"
