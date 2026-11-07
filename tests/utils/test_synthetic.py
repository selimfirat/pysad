import numpy as np
import pytest


def test_generate_stream_shapes_and_labels():
    from pysad.utils import generate_stream

    X, y = generate_stream(
        num_instances=500,
        num_features=3,
        num_points=5,
        num_bursts=2,
        burst_length=10,
        num_drifts=1,
        drift_length=40,
        random_state=0,
    )

    assert X.shape == (500, 3)
    assert X.dtype == np.float64
    assert y.shape == (500,)
    assert set(np.unique(y)) == {0, 1}
    assert y.sum() == 5 + 2 * 10 + 40
    assert y[:50].sum() == 0


def test_generate_stream_random_state():
    from pysad.utils import generate_stream

    X1, y1 = generate_stream(random_state=0)
    X2, y2 = generate_stream(random_state=0)
    X3, y3 = generate_stream(random_state=1)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    assert not np.array_equal(y1, y3)


def test_generate_stream_anomalies_are_far_from_normal():
    from pysad.utils import generate_stream

    X, y = generate_stream(num_drifts=0, magnitude=6.0, random_state=0)
    distances = np.linalg.norm(X, axis=1)

    assert distances[y == 1].min() > np.median(distances[y == 0])


def test_generate_stream_too_many_anomalies():
    from pysad.utils import generate_stream

    with pytest.raises(ValueError, match="do not fit"):
        generate_stream(num_instances=100, random_state=0)
