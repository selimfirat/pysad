"""Compare optional kernels with the default detector outputs."""

import pickle

import numpy as np
import pytest

from pysad import _kernels
from pysad.models import LODA, StreamKMPlusPlus


@pytest.mark.parametrize("model_class", [LODA, StreamKMPlusPlus])
def test_numpy_default_and_missing_extension(model_class, monkeypatch):
    """An unavailable extension keeps the same scores and fitted outputs."""
    monkeypatch.setattr(_kernels, "loda_neg_log_densities", None, raising=False)
    monkeypatch.setattr(_kernels, "streamkm_lloyd", None, raising=False)
    rng = np.random.RandomState(18)
    stream = rng.normal(size=(25, 3))
    default = model_class(random_state=7)
    fallback = model_class(random_state=7, use_rust=True)
    np.testing.assert_array_equal(default.fit_score(stream), fallback.fit_score(stream))
    restored = pickle.loads(pickle.dumps(fallback))
    assert restored.score_partial(stream[0]) == fallback.score_partial(stream[0])


@pytest.mark.parametrize("model_class", [LODA, StreamKMPlusPlus])
def test_numpy_default_ignores_installed_kernels(model_class, monkeypatch):
    def fail(*args):
        raise AssertionError("The default path must use NumPy.")

    monkeypatch.setattr(_kernels, "loda_neg_log_densities", fail, raising=False)
    monkeypatch.setattr(_kernels, "streamkm_lloyd", fail, raising=False)
    stream = np.random.RandomState(4).normal(size=(20, 3))
    scores = model_class(random_state=0).fit_score(stream)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("model_class", [LODA, StreamKMPlusPlus])
def test_native_stream_equivalence(model_class):
    pytest.importorskip("pysad_kernels")
    rng = np.random.RandomState(12)
    stream = rng.normal(size=(65, 4))
    stream[10:15] = stream[0]
    stream[32] *= 100
    params = {"window_size": 9} if model_class is LODA else {"coreset_size": 8, "n_max": 8}
    reference = model_class(random_state=2, **params)
    native = model_class(random_state=2, use_rust=True, **params)
    np.testing.assert_allclose(
        native.fit_score(stream), reference.fit_score(stream), rtol=1e-10, atol=1e-10
    )
    queries = rng.normal(size=(5, 4))
    np.testing.assert_allclose(native.score(queries), reference.score(queries), atol=1e-10)
    if model_class is LODA:
        np.testing.assert_allclose(
            native.explain_partial(queries[0]), reference.explain_partial(queries[0]), atol=1e-10
        )
    restored = pickle.loads(pickle.dumps(native))
    np.testing.assert_allclose(restored.fit_score(queries), native.fit_score(queries), atol=1e-10)


def test_native_loda_density_edges():
    kernels = pytest.importorskip("pysad_kernels")
    # Zero width, an empty bin, both range edges, and a NaN bin index.
    indices = np.array([0.0, 1.0, -1.0, 2.0, np.nan])
    histograms = np.array([[4.0, 0.0]] * 5)
    widths = np.array([0.0, 0.5, 0.5, 0.5, 0.5])
    result = kernels.loda_neg_log_densities(indices, histograms, widths, 4)
    np.testing.assert_allclose(result, [0.0, np.log(3), np.log(3), np.log(3), np.log(3)])


def test_native_lloyd_weights_ties_and_empty_centers():
    kernels = pytest.importorskip("pysad_kernels")
    points = np.array([[0.0], [2.0], [10.0], [100.0]])
    weights = np.array([1.0, 3.0, 2.0, 0.0])
    centers = np.array([[0.0], [4.0], [50.0], [60.0]])
    result = kernels.streamkm_lloyd(points, weights, centers, 1)
    # The point at 2 ties and enters the first cluster. Empty centers stay in place.
    np.testing.assert_array_equal(result, [[1.5], [10.0], [50.0], [60.0]])
    np.testing.assert_array_equal(centers, [[0.0], [4.0], [50.0], [60.0]])


def test_native_kernels_accept_strided_inputs():
    kernels = pytest.importorskip("pysad_kernels")
    points = np.array([[0.0, 9.0], [2.0, 9.0], [10.0, 9.0]])[:, ::2]
    result = kernels.streamkm_lloyd(points, np.ones(3), points[::2], 2)
    np.testing.assert_array_equal(result, [[1.0], [10.0]])
    result = kernels.loda_neg_log_densities(
        np.array([0.0, 9.0, 1.0])[::2], np.ones((2, 4))[:, ::2], np.ones(4)[::2], 2
    )
    np.testing.assert_allclose(result, np.log(2))
