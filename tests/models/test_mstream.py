import numpy as np
import pytest


@pytest.mark.parametrize("tick_size", [1, 3])
def test_mstream_scores_repeated_records_by_the_chi_squared_statistic(tick_size):
    from pysad.models import MStream

    alpha, num_features, num_records = 0.5, 2, 10
    X = np.ones((num_records, num_features))
    scores = MStream(alpha=alpha, tick_size=tick_size, random_state=0).fit_score(X)

    # Every record shares one bucket in each feature and in the sketch, so the num_features + 1
    # counts are equal: s counts all records, and a decays by alpha at the start of each tick.
    expected, a = [], 0.0
    for i in range(num_records):
        t = i // tick_size + 1
        a = (alpha * a if i % tick_size == 0 else a) + 1.0
        s = i + 1
        expected.append(
            0.0 if t == 1 else (num_features + 1) * (a - s / t) ** 2 * t**2 / (s * (t - 1))
        )
    np.testing.assert_allclose(scores, expected)


def test_mstream_scores_a_burst_of_repeated_records_above_the_stream():
    from pysad.models import MStream

    rng = np.random.default_rng(0)
    stream = rng.uniform(size=(500, 3))
    burst = np.full((20, 3), 0.5)
    scores = MStream(random_state=0).fit_score(np.vstack([stream, burst]))

    assert scores[-10:].min() > scores[100:500].max()


def test_mstream_log_transform_needs_values_above_minus_one():
    from pysad.models import MStream

    X = np.array([0.0, -1.0])
    with pytest.raises(ValueError, match="log_transform=False"):
        MStream().fit_partial(X)
    assert MStream(log_transform=False).fit_partial(X).fit_score_partial(X) >= 0.0


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_mstream_rejects_an_alpha_outside_zero_and_one(alpha):
    from pysad.models import MStream

    with pytest.raises(ValueError, match="alpha"):
        MStream(alpha=alpha)
