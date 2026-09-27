def _verify_initial_window_reproducibility(window_X):
    import numpy as np
    from pysad.models import HalfSpaceTrees
    from pysad.utils import fix_seed

    # Generate test data with fixed seed
    fix_seed(42)
    initial_window_X = np.random.rand(10, 2)
    test_X = np.random.rand(5, 2)

    # Model with initial window
    fix_seed(42)
    model_with_window = HalfSpaceTrees(feature_mins=[0, 0], feature_maxes=[1, 1],
                                       initial_window_X=window_X(initial_window_X))

    # Model without initial window, then fit with same data
    fix_seed(42)
    model_without_window = HalfSpaceTrees(feature_mins=[0, 0], feature_maxes=[1, 1])
    model_without_window.fit(initial_window_X)

    # Both models should produce identical scores on test data
    scores_with_window = np.array([model_with_window.score_partial(x) for x in test_X])
    scores_without_window = np.array([model_without_window.score_partial(x) for x in test_X])

    np.testing.assert_array_equal(scores_with_window, scores_without_window)


def test_half_space_trees_with_numpy_initial_window():
    _verify_initial_window_reproducibility(lambda x: x)


def test_half_space_trees_with_list_initial_window():
    _verify_initial_window_reproducibility(lambda x: x.tolist())
