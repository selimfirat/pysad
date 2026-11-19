import numpy as np
import pytest

from pysad.transform.stream_generators import SlowPoisonStream


def _normal_stream(num_instances: int = 100, num_features: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).standard_normal((num_instances, num_features))


def test_slow_poison_marks_injected_points():
    gen = SlowPoisonStream(random_state=1, poison_rate=0.5, warmup=0, magnitude=5.0)
    X = _normal_stream(200, 5, seed=2)
    labels: list[bool] = []
    outs: list[np.ndarray] = []
    for x in X:
        out = gen.fit_transform_partial(x)
        labels.append(gen.is_anomaly)
        outs.append(out)
    labels_arr = np.array(labels)
    outs_arr = np.array(outs)
    assert labels_arr.sum() > 0
    np.testing.assert_array_equal(outs_arr[~labels_arr], X[~labels_arr])
    assert np.all(np.any(~np.isclose(outs_arr[labels_arr], X[labels_arr]), axis=1))


def test_slow_poison_rate_zero_never_poisons():
    gen = SlowPoisonStream(random_state=0, poison_rate=0.0, warmup=0)
    X = _normal_stream(50, 3, seed=0)
    for x in X:
        gen.fit_transform_partial(x)
        assert gen.is_anomaly is False


def test_slow_poison_rate_one_always_poisons():
    gen = SlowPoisonStream(random_state=0, poison_rate=1.0, warmup=0)
    X = _normal_stream(50, 3, seed=0)
    for x in X:
        gen.fit_transform_partial(x)
        assert gen.is_anomaly is True


def test_slow_poison_is_reproducible():
    X = _normal_stream(100, 3, seed=0)

    def run() -> tuple[np.ndarray, np.ndarray]:
        gen = SlowPoisonStream(random_state=2, poison_rate=0.3, warmup=10, magnitude=4.0)
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


def test_slow_poison_output_matches_input_shape_and_is_finite():
    gen = SlowPoisonStream(random_state=0, poison_rate=0.5, warmup=0, magnitude=5.0)
    out = gen.fit_transform_partial(np.ones(4))
    assert out.shape == (4,)
    assert np.all(np.isfinite(out))


def test_slow_poison_fit_partial_is_normal_and_returns_self():
    gen = SlowPoisonStream(random_state=0)
    x = np.ones(3)
    assert gen.fit_partial(x) is gen
    assert gen.is_anomaly is False


def test_slow_poison_rejects_feature_mismatch():
    gen = SlowPoisonStream(random_state=0)
    gen.fit_partial(np.ones(3))
    with pytest.raises(ValueError):
        gen.transform_partial(np.ones(4))


@pytest.mark.parametrize("poison_rate", [-0.1, 1.5])
def test_slow_poison_rejects_bad_poison_rate(poison_rate: float) -> None:
    with pytest.raises(ValueError):
        SlowPoisonStream(poison_rate=poison_rate)


def test_slow_poison_rejects_bad_types() -> None:
    with pytest.raises(TypeError):
        SlowPoisonStream(poison_rate="high")
    with pytest.raises(TypeError):
        SlowPoisonStream(magnitude="high")
    with pytest.raises(TypeError):
        SlowPoisonStream(warmup="high")
    with pytest.raises(ValueError):
        SlowPoisonStream(magnitude=-1.0)
    with pytest.raises(ValueError):
        SlowPoisonStream(warmup=-1)
