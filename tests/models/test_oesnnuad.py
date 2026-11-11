import numpy as np
import pytest

from pysad.models import OeSNNUAD
from pysad.utils import fix_seed


def test_scores_zero_until_the_window_is_full():
    """The first window_size values only fill the window, so they score 0.0."""
    X = np.sin(np.arange(40) / 3.0).reshape(-1, 1)

    fix_seed(0)
    scores = OeSNNUAD(window_size=5).fit_score(X)

    np.testing.assert_array_equal(scores[:5], 0.0)
    assert np.isfinite(scores).all()
    # The model scores each later value on its own, so they are not all equal.
    assert len(set(scores[5:])) > 1


def test_a_spike_scores_above_the_normal_values():
    """A value that is far from the values that the model learned gets a score above epsilon."""
    rng = np.random.RandomState(0)
    X = (np.sin(2 * np.pi * np.arange(300) / 24) + rng.normal(0.0, 0.1, 300)).reshape(-1, 1)
    X[150] += 5.0

    fix_seed(1)
    scores = OeSNNUAD(window_size=100, epsilon=4.0).fit_score(X)

    assert scores[150] >= 4.0
    assert scores[150] > np.median(scores[100:])
    # Only the spike and its direct neighbours reach the flag of the paper.
    assert (scores[100:] >= 4.0).mean() < 0.1


def test_no_output_neuron_fires_gives_the_largest_score():
    """The repository is empty while the window fills, so no output neuron fires for the first
    scored value. The paper sets the error of such a value to +infinity."""
    model = OeSNNUAD(window_size=4)
    model.fit(np.zeros((4, 1)))

    assert model.score_partial(np.array([1.0])) == np.finfo(np.float64).max


def test_score_partial_does_not_change_the_model():
    """A score comes from the state before the value, so scoring a value twice changes nothing."""
    rng = np.random.RandomState(1)
    X = rng.rand(40, 1)
    model = OeSNNUAD(window_size=8)

    fix_seed(0)
    streamed = []
    for xi in X:
        model.score_partial(xi)
        streamed.append(model.fit_score_partial(xi))

    fix_seed(0)
    np.testing.assert_array_equal(streamed, OeSNNUAD(window_size=8).fit_score(X))


def test_the_same_seed_gives_the_same_scores():
    """The output values of the neurons are random, and the seed fixes them."""
    X = np.random.RandomState(2).rand(200, 1) * 5.0

    fix_seed(0)
    first = OeSNNUAD(window_size=20, epsilon=3.0, random_state=7).fit_score(X)
    fix_seed(1)
    second = OeSNNUAD(window_size=20, epsilon=3.0, random_state=7).fit_score(X)
    fix_seed(2)
    third = OeSNNUAD(window_size=20, epsilon=3.0, random_state=8).fit_score(X)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_the_repository_stays_bounded():
    """The repository holds at most num_output_neurons neurons, whatever the length of the stream."""
    X = np.random.RandomState(3).rand(400, 1)

    model = OeSNNUAD(window_size=20, num_output_neurons=4)
    model.fit(X)

    assert model.num_neurons_ == 4


def test_similar_values_merge_into_one_output_neuron():
    """Values with the same encoding give the same weights, so their neurons merge."""
    model = OeSNNUAD(window_size=5)
    model.fit(np.full((60, 1), 2.0))

    assert model.num_neurons_ == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 1},
        {"window_size": 10, "num_input_neurons": 2},
        {"window_size": 10, "num_output_neurons": 0},
        {"window_size": 10, "sim": -0.1},
        {"window_size": 10, "mod": 0.0},
        {"window_size": 10, "mod": 1.0},
        {"window_size": 10, "c": 0.0},
        {"window_size": 10, "c": 1.5},
        {"window_size": 10, "xi": -0.1},
        {"window_size": 10, "xi": 1.2},
        {"window_size": 10, "epsilon": -1.0},
    ],
)
def test_rejects_parameters_outside_the_paper_range(kwargs):
    with pytest.raises(ValueError):
        OeSNNUAD(**kwargs)


@pytest.mark.parametrize("kwargs", [{"window_size": 4.5}, {"num_input_neurons": 2.5}])
def test_rejects_parameters_that_are_not_ints(kwargs):
    with pytest.raises(TypeError):
        OeSNNUAD(**kwargs)
