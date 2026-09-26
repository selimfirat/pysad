

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


def test_fit_draws_projection_used_by_transform():
    import numpy as np
    from pysad.transform.projection import GaussianRandomProjector, SparseRandomProjector

    X = np.random.RandomState(0).rand(20, 100)

    for projector_cls in [GaussianRandomProjector, SparseRandomProjector]:
        projector = projector_cls(num_components=10).fit(X)
        components = projector._components

        assert projector.transform(X).shape == (20, 10)
        assert projector._components is components


def test_auto_components_are_sized_from_the_batch():
    import numpy as np
    import pytest
    from sklearn.random_projection import johnson_lindenstrauss_min_dim
    from pysad.transform.projection import GaussianRandomProjector, SparseRandomProjector

    X = np.random.RandomState(0).rand(50, 500)
    num_components = johnson_lindenstrauss_min_dim(50, eps=0.5)

    for projector_cls in [GaussianRandomProjector, SparseRandomProjector]:
        assert projector_cls(eps=0.5).fit_transform(X).shape == (50, num_components)
        assert projector_cls(eps=0.5).fit(X).transform_partial(X[0]).shape == (num_components,)

        with pytest.raises(ValueError, match="num_components='auto'"):
            projector_cls(eps=0.5).transform_partial(X[0])
