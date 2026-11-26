import pytest


def test_projection_is_consistent_across_instances():
    import numpy as np

    from pysad.transform.projection import GaussianRandomProjector, SparseRandomProjector

    X = np.random.RandomState(0).rand(20, 100)

    for projector in [
        GaussianRandomProjector(num_components=10),
        SparseRandomProjector(num_components=10),
    ]:
        projected_X = projector.fit_transform(X)

        assert np.allclose(projector.transform_partial(X[0]), projected_X[0])
        assert np.allclose(projector.transform_partial(2 * X[0]), 2 * projected_X[0])


def test_auto_components_are_sized_from_the_batch():
    import numpy as np
    import pytest
    from sklearn.random_projection import johnson_lindenstrauss_min_dim

    from pysad.transform.projection import GaussianRandomProjector, SparseRandomProjector

    X = np.random.RandomState(0).rand(50, 500)
    num_components = johnson_lindenstrauss_min_dim(50, eps=0.5)

    for projector_cls in [GaussianRandomProjector, SparseRandomProjector]:
        assert projector_cls(eps=0.5).fit_transform(X).shape == (50, num_components)
        assert projector_cls(eps=0.5).transform(X).shape == (50, num_components)
        assert projector_cls(eps=0.5).fit(X).transform_partial(X[0]).shape == (num_components,)

        with pytest.raises(ValueError, match="num_components='auto'"):
            projector_cls(eps=0.5).transform_partial(X[0])


@pytest.mark.parametrize("projector_name", ["GaussianRandomProjector", "SparseRandomProjector"])
def test_projector_random_state(projector_name):
    import numpy as np

    import pysad.transform.projection
    from pysad.utils import fix_seed

    projector_cls = getattr(pysad.transform.projection, projector_name)
    x = np.arange(20, dtype=np.float64)

    def project(**kwargs):
        return projector_cls(num_components=5, **kwargs).fit_transform_partial(x)

    fix_seed(3)
    default = project()

    np.testing.assert_array_equal(default, project(random_state=3))
    np.testing.assert_array_equal(default, project(random_state=np.random.RandomState(3)))
    assert not np.array_equal(default, project(random_state=4))

    # A seeded projector leaves the global state alone.
    fix_seed(3)
    expected = np.random.rand(3)
    fix_seed(3)
    project(random_state=4)
    np.testing.assert_array_equal(np.random.rand(3), expected)
