import numpy as np
import pytest

from pysad.transform.stream_generators import StealthBurstStream


def _normal_stream(num_instances: int = 100, num_features: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).standard_normal((num_instances, num_features))


def test_stealth_burst_marks_burst_points():
    gen = StealthBurstStream(random_state=0, burst_every=10, burst_length=3, magnitude=4.0)
    X = _normal_stream(40, 3, seed=0)
    labels: list[bool] = []
    for x in X:
        gen.fit_transform_partial(x)
        labels.append(gen.is_anomaly)
    labels_arr = np.array(labels)
    assert labels_arr[:10].sum() == 0
    assert labels_arr[10:13].all()
    assert labels_arr[13:20].sum() == 0
    assert labels_arr[20:23].all()


def test_stealth_burst_normal_points_unchanged():
    gen = StealthBurstStream(random_state=0, burst_every=10, burst_length=3, magnitude=4.0)
    X = _normal_stream(40, 3, seed=0)
    labels: list[bool] = []
    outs: list[np.ndarray] = []
    for x in X:
        out = gen.fit_transform_partial(x)
        labels.append(gen.is_anomaly)
        outs.append(out)
    labels_arr = np.array(labels)
    outs_arr = np.array(outs)
    np.testing.assert_array_equal(outs_arr[~labels_arr], X[~labels_arr])
    assert np.any(~np.isclose(outs_arr[labels_arr], X[labels_arr]))


def test_stealth_burst_is_reproducible():
    X = _normal_stream(100, 3, seed=0)

    def run() -> tuple[np.ndarray, np.ndarray]:
        gen = StealthBurstStream(random_state=2, burst_every=12, burst_length=4, magnitude=3.0)
        labels: list[bool] = []
        outs: list[np.ndarray] = []
        for x in X:
            out = gen.fit_transform_partial(x)
            labels.append(gen.is_anomaly)
            outs.append(out)
        return np.array(labels), np.array(outs)

    labels1, outs1 = run()
    labels2, outs2 = run()
    assert np.array_equal(labels1, labels2)
    np.testing.assert_array_equal(outs1, outs2)


def test_stealth_burst_output_matches_input_shape_and_is_finite():
    gen = StealthBurstStream(random_state=0, burst_every=10, burst_length=3, magnitude=4.0)
    out = gen.fit_transform_partial(np.ones(4))
    assert out.shape == (4,)
    assert np.all(np.isfinite(out))


def test_stealth_burst_fit_partial_is_normal_and_returns_self():
    gen = StealthBurstStream(random_state=0)
    x = np.ones(3)
    assert gen.fit_partial(x) is gen
    assert gen.is_anomaly is False


def test_stealth_burst_rejects_feature_mismatch():
    gen = StealthBurstStream(random_state=0)
    gen.fit_partial(np.ones(3))
    with pytest.raises(ValueError):
        gen.transform_partial(np.ones(4))


@pytest.mark.parametrize("burst_every", [0, -1])
def test_stealth_burst_rejects_bad_burst_every(burst_every: int) -> None:
    with pytest.raises(ValueError):
        StealthBurstStream(burst_every=burst_every)


@pytest.mark.parametrize("burst_length", [0, -1])
def test_stealth_burst_rejects_bad_burst_length(burst_length: int) -> None:
    with pytest.raises(ValueError):
        StealthBurstStream(burst_length=burst_length)


def test_stealth_burst_rejects_bad_types() -> None:
    with pytest.raises(TypeError):
        StealthBurstStream(burst_every="high")
    with pytest.raises(TypeError):
        StealthBurstStream(burst_length="high")
    with pytest.raises(TypeError):
        StealthBurstStream(magnitude="high")
    with pytest.raises(ValueError):
        StealthBurstStream(magnitude=-1.0)
