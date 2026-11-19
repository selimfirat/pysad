import numpy as np
import pytest


def test_first_distinct_instances_become_centers():
    from pysad.models import OnlineKMeans

    model = OnlineKMeans(num_clusters=2, random_state=0)
    scores = model.fit_score(np.array([[0.0], [0.0], [3.0], [1.0]]))

    # Each instance is scored by its squared distance to the distinct instances before it.
    np.testing.assert_array_equal(scores, [0.0, 0.0, 9.0, 1.0])
    assert model.score_partial(np.array([2.0])) == 1.0
    assert model.score_partial(np.array([3.0])) == 0.0


def _opens_center(model, x):
    model.fit_partial(x)
    return model.score_partial(x) == 0.0


@pytest.mark.parametrize("seed", range(5))
def test_far_instance_always_opens_a_center(seed):
    from pysad.models import OnlineKMeans

    # The initial centers 0 and 1 give w* = 1 / 2 and a facility cost of w* / k = 1 / 2.
    model = OnlineKMeans(num_clusters=1, random_state=seed).fit(np.array([[0.0], [1.0]]))

    assert _opens_center(model, np.array([10.0]))


def test_instance_opens_a_center_with_probability_of_its_squared_distance_over_the_facility_cost():
    from pysad.models import OnlineKMeans

    # The initial centers 0 and 2 give a facility cost of 2, and 1 is at squared distance 1 from both.
    opened = [
        _opens_center(
            OnlineKMeans(num_clusters=1, random_state=seed).fit(np.array([[0.0], [2.0]])),
            np.array([1.0]),
        )
        for seed in range(2000)
    ]

    assert np.mean(opened) == pytest.approx(0.5, abs=0.03)


def test_facility_cost_doubles_after_enough_centers_open():
    from pysad.models import OnlineKMeans

    def model_after(num_far, seed):
        # The facility cost starts at 1 / 2, and each far instance opens a center.
        model = OnlineKMeans(num_clusters=1, random_state=seed).fit(np.array([[0.0], [1.0]]))
        return model.fit(10.0 * np.arange(1, num_far + 1).reshape(-1, 1))

    # With k = 1, the 11th opened center is the first to reach 3 * k * (1 + log(n)), at n = 13.
    x = np.array([-np.sqrt(0.5)])
    before = [_opens_center(model_after(10, seed), x) for seed in range(1000)]
    after = [_opens_center(model_after(11, seed), x) for seed in range(1000)]

    assert all(before)
    assert np.mean(after) == pytest.approx(0.5, abs=0.04)


def test_outlier_gets_the_highest_score():
    from pysad.models import OnlineKMeans

    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(size=(300, 2)), [[8.0, 8.0]]])

    scores = OnlineKMeans(random_state=0).fit_score(X)

    assert scores.argmax() == len(X) - 1


def test_new_center_replaces_the_oldest_when_max_centers_is_reached():
    from pysad.models import OnlineKMeans

    # The initial centers 0 and 1 fill max_centers, and the far instance 10 always opens a center.
    model = OnlineKMeans(num_clusters=1, max_centers=2, random_state=0)
    model.fit(np.array([[0.0], [1.0], [10.0]]))

    assert model.score_partial(np.array([10.0])) == 0.0
    assert model.score_partial(np.array([1.0])) == 0.0
    assert model.score_partial(np.array([0.0])) == 1.0


def test_max_centers_must_exceed_num_clusters():
    from pysad.models import OnlineKMeans

    with pytest.raises(ValueError, match="max_centers"):
        OnlineKMeans(num_clusters=10, max_centers=10)
