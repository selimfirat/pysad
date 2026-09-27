import numpy as np


def test_streamhash_projector(test_path):
    from sklearn.utils import shuffle
    from pysad.utils import Data
    import os
    from pysad.transform.projection import StreamhashProjector

    for num_components in [2, 50, 250]:
        data = Data(os.path.join(test_path, "../../../examples/data"))

        X_all, y_all = data.get_data("arrhythmia.mat")
        X_all, y_all = shuffle(X_all, y_all)

        projector = StreamhashProjector(num_components=num_components)

        projected_X = projector.fit_transform(X_all)

        assert projected_X.shape == (X_all.shape[0], num_components)


def _reference_projection_matrix(projector, ndim):
    """Builds the projection matrix with the original per-entry formula, independently of the cache."""
    import mmh3

    def hash_string(k, s):
        hash_value = int(mmh3.hash(s, signed=False, seed=int(k))) / (2.0 ** 32 - 1)
        density = projector.density
        if hash_value <= density / 2.0:
            return -1 * projector.constant
        elif hash_value <= density:
            return projector.constant
        else:
            return 0

    feature_names = [str(i) for i in range(ndim)]
    return np.array([[hash_string(k, f) for f in feature_names] for k in projector.keys])


def test_streamhash_projector_cached_matrix_matches_reference():
    from pysad.transform.projection import StreamhashProjector

    num_components = 10
    ndim = 7
    rng = np.random.RandomState(61)

    projector = StreamhashProjector(num_components=num_components)
    R_ref = _reference_projection_matrix(projector, ndim)

    for _ in range(5):
        x = rng.rand(ndim)

        expected = x @ R_ref.T
        actual = projector.transform_partial(x)

        assert np.array_equal(actual, expected)


def test_streamhash_projector_rebuilds_matrix_on_ndim_change():
    from pysad.transform.projection import StreamhashProjector

    num_components = 10
    rng = np.random.RandomState(61)

    projector = StreamhashProjector(num_components=num_components)

    x_narrow = rng.rand(5)
    projector.transform_partial(x_narrow)

    x_wide = rng.rand(9)
    R_ref_wide = _reference_projection_matrix(projector, 9)
    actual = projector.transform_partial(x_wide)

    assert actual.shape == (num_components,)
    assert np.array_equal(actual, x_wide @ R_ref_wide.T)
