import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

M = 20


def _series():
    """A noisy sine of period 50 with a nearly flat stretch at 700 and a ramp at 1000."""
    rng = np.random.default_rng(0)
    values = np.sin(2 * np.pi * np.arange(1500) / 50) + 0.1 * rng.normal(size=1500)
    values[700:720] = 0.3 + 0.01 * rng.normal(size=20)
    values[1000:1020] += np.linspace(0.0, 2.0, 20)
    return values


def _znormalize(x):
    return (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)


def _left_matrix_profile(values, m, lag, first):
    """The exact left matrix profile value of the subsequence that ends at each value from `first` on, by brute force.

    The neighbors of the subsequence that starts at i end at or before i and start at or after i - lag.
    """
    profile = np.zeros(values.size)
    for end in range(first, values.size):
        i = end - m + 1
        query = _znormalize(values[i : end + 1])
        neighbors = sliding_window_view(values[max(i - lag, 0) : i + 1], m)
        profile[end] = np.linalg.norm(query - _znormalize(neighbors), axis=1).min()

    return profile


# A history shorter than the probationary period keeps fewer values than the model learns from.
@pytest.mark.parametrize("history_size", [10000, 100, M - 1])
def test_scores_are_bounded_by_the_exact_left_matrix_profile(history_size):
    from pysad.models import DAMP

    values = _series()
    scores = DAMP(window_length=M, history_size=history_size).fit_score(values.reshape(-1, 1))
    exact = _left_matrix_profile(values, M, history_size, first=4 * M)

    # Each score lies between the exact value and the best-so-far score, and is exact when it is
    # a new best-so-far score.
    best_so_far = np.maximum.accumulate(exact)
    np.testing.assert_array_less(exact - 1e-9, scores)
    np.testing.assert_array_less(scores, best_so_far + 1e-9)
    new_best = exact == best_so_far
    np.testing.assert_allclose(scores[new_best], exact[new_best], rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(scores[: 4 * M], 0.0)


def test_flat_stretch_gets_the_highest_score():
    from pysad.models import DAMP

    scores = DAMP(window_length=M).fit_score(_series().reshape(-1, 1))

    # The highest score ends a subsequence that overlaps the flat stretch, the top left-discord.
    assert 700 <= scores.argmax() < 720 + M


def test_score_partial_does_not_change_the_model():
    from pysad.models import DAMP

    values = _series()[:200]
    model = DAMP(window_length=M, history_size=100).fit(values[:150].reshape(-1, 1))
    scores = model.score(values[150:].reshape(-1, 1))

    np.testing.assert_array_equal(model.score(values[150:].reshape(-1, 1)), scores)


@pytest.mark.parametrize(
    ("params", "error"),
    [
        ({"window_length": 1}, ValueError),
        ({"window_length": 2.0}, TypeError),
        ({"window_length": 10, "history_size": 8}, ValueError),
        ({"window_length": 10, "probationary_period": 17}, ValueError),
    ],
)
def test_invalid_parameters(params, error):
    from pysad.models import DAMP

    with pytest.raises(error):
        DAMP(**params)


def test_rejects_multivariate_instances():
    from pysad.models import DAMP

    with pytest.raises(ValueError, match="univariate"):
        DAMP(window_length=4).fit_partial(np.array([1.0, 2.0]))
