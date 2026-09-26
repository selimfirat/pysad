

def test_gaussian_random_projector(test_path):
    from pysad.transform.projection import GaussianRandomProjector

    for num_components in [2, 50, 250]:

        projector = GaussianRandomProjector(num_components=num_components)

        helper_test_projector(test_path, projector, num_components)


def test_sparse_random_projector(test_path):
    from pysad.transform.projection import SparseRandomProjector

    for num_components in [2, 50, 250]:

        projector = SparseRandomProjector(num_components=num_components)

        helper_test_projector(test_path, projector, num_components)


def helper_test_projector(test_path, projector, num_components):
    import os
    from sklearn.utils import shuffle
    from pysad.utils import Data

    data = Data(os.path.join(test_path, "../../../examples/data"))

    X_all, y_all = data.get_data("arrhythmia.mat")
    X_all, y_all = shuffle(X_all, y_all)
    projected_X = projector.fit_transform(X_all)

    assert projected_X.shape == (X_all.shape[0], num_components)


def test_projection_is_consistent_across_instances():
    import numpy as np
    from pysad.transform.projection import GaussianRandomProjector, SparseRandomProjector

    X = np.random.RandomState(0).rand(20, 100)

    for projector in [GaussianRandomProjector(num_components=10), SparseRandomProjector(num_components=10)]:
        projected_X = projector.fit_transform(X)

        assert np.allclose(projector.transform_partial(X[0]), projected_X[0])
        assert np.allclose(projector.transform_partial(2 * X[0]), 2 * projected_X[0])


def test_sparse_projector_keeps_components_sparse():
    import numpy as np
    from scipy.sparse import issparse
    from pysad.transform.projection import SparseRandomProjector

    projector = SparseRandomProjector(num_components=10)
    projector.fit_transform(np.random.RandomState(0).rand(5, 1000))

    assert issparse(projector._components)
