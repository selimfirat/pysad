import pickle
import warnings

import numpy as np


def test_murmurhash3_x86_32_matches_mmh3():
    """The vectorized hash must equal mmh3.hash(str(j), signed=False, seed=k) bit for bit."""
    import mmh3

    from pysad.transform.projection.streamhash_projector import _murmurhash3_x86_32

    feature_indices = list(range(20001)) + [123456, 9999999]
    seeds = list(range(50)) + [2**31 - 1]

    seeds_arr = np.array(seeds, dtype=np.uint32)
    lengths = {}
    for i in feature_indices:
        lengths.setdefault(len(str(i)), []).append(i)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        for indices in lengths.values():
            byte_matrix = np.array([[ord(c) for c in str(i)] for i in indices], dtype=np.uint8)
            actual = _murmurhash3_x86_32(seeds_arr, byte_matrix)

            for row, seed in enumerate(seeds):
                for col, i in enumerate(indices):
                    expected = mmh3.hash(str(i), signed=False, seed=seed)
                    assert actual[row, col] == expected, (seed, i)


def test_streamhash_projector(test_path):
    import os

    from sklearn.utils import shuffle

    from pysad.transform.projection import StreamhashProjector
    from pysad.utils import Data

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
        hash_value = int(mmh3.hash(s, signed=False, seed=int(k))) / (2.0**32 - 1)
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


def test_build_projection_matrix_matches_reference_for_various_shapes():
    """Full matrix build (vectorized) must equal the per-entry mmh3 formula across digit-length groups."""
    from pysad.transform.projection import StreamhashProjector

    for num_components, ndim in [(2, 1), (50, 123), (250, 274), (3, 1234)]:
        projector = StreamhashProjector(num_components=num_components)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            actual = projector._get_projection_matrix(ndim)

        expected = _reference_projection_matrix(projector, ndim)

        assert actual.shape == (num_components, ndim)
        assert np.array_equal(actual, expected)


def test_streamhash_projector_loads_old_pickle_without_cache_attrs():
    """Instances pickled by older pysad versions lack `_R`/`_R_ndim` in their __dict__."""
    from pysad.transform.projection import StreamhashProjector

    num_components = 10
    rng = np.random.RandomState(61)
    x = rng.rand(7)

    fresh_projector = StreamhashProjector(num_components=num_components)
    expected = fresh_projector.transform_partial(x)

    old_projector = StreamhashProjector(num_components=num_components)
    del old_projector.__dict__["_R"]
    del old_projector.__dict__["_R_ndim"]

    restored_projector = pickle.loads(pickle.dumps(old_projector))

    actual = restored_projector.transform_partial(x)

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
