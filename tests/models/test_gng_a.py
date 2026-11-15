import numpy as np
import pytest


def test_creation_waits_for_recent_wins():
    """Missing history cannot count as wins and trigger early creation."""
    from pysad.models import GNGA

    model = GNGA(k=1, random_state=0)
    model.fit(np.array([[0.0], [4.0], [1.0], [2.0]]))

    # At most two wins give fq <= 2/10, below the retarder 1/4.
    assert model.num_neurons_ == 2


def test_error_ranked_movement_and_anomaly_margin():
    """The published learning rates and mean edge length determine the score."""
    from pysad.models import GNGA

    model = GNGA(max_neurons=2, random_state=0)
    model.fit(np.array([[0.0], [4.0], [1.0]]))

    # The winner moves by 1/2. Its neighbor moves by 1/3.
    np.testing.assert_allclose(model.neurons_, [[0.5], [3.0]])
    assert model.score_partial(np.array([6.0])) == pytest.approx(0.5)
    assert model.score_partial(np.array([0.5])) == pytest.approx(-2.5)


def test_coincident_prototypes_keep_finite_distance_scores():
    """A zero-length edge cannot make the anomaly margin divide by zero."""
    from pysad.models import GNGA

    model = GNGA(random_state=0).fit(np.zeros((2000, 2)))

    assert model.score_partial(np.zeros(2)) == 0.0
    assert model.score_partial(np.array([3.0, 4.0])) == 5.0
    assert np.isfinite(model.threshold_)


def test_graph_grows_within_capacity_and_follows_drift():
    """A bounded graph learns new regions and forgets stale prototypes."""
    from pysad.models import GNGA

    rng = np.random.RandomState(3)
    model = GNGA(max_neurons=6, random_state=5)
    sizes = []
    for point in rng.normal(size=(2000, 2)):
        model.fit_partial(point)
        sizes.append(model.num_neurons_)
    before = model.score_partial(np.array([30.0, 30.0]))
    for point in rng.normal(30.0, 1.0, size=(2000, 2)):
        model.fit_partial(point)
        sizes.append(model.num_neurons_)

    assert max(sizes) == 6
    assert np.any(np.diff(sizes[1:]) < 0)
    assert model.score_partial(np.array([30.0, 30.0])) < before / 10
    assert model.score_partial(np.zeros(2)) > 20.0
    assert np.all(model.neurons_ > 20.0)
