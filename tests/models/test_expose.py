import numpy as np
import pytest


@pytest.mark.parametrize("decay", [None, 0.1])
def test_scores_approximate_the_normalized_expected_similarity(decay):
    from pysad.models import EXPoSE

    rng = np.random.RandomState(0)
    X, z = rng.normal(size=(50, 3)), rng.normal(size=3)
    gamma = 0.3
    model = EXPoSE(num_random_features=100000, gamma=gamma, decay=decay, random_state=0).fit(X)

    # The weights of the instances in the embedding: equal for the running mean, and (1 - decay)
    # to the power of their age for gradual forgetting, where the first instance starts at 1.
    if decay is None:
        weights = np.full(len(X), 1.0 / len(X))
    else:
        weights = decay * (1.0 - decay) ** np.arange(len(X) - 1, -1, -1)
        weights[0] = (1.0 - decay) ** (len(X) - 1)
    kernel = np.exp(-gamma * ((X[:, None] - X[None]) ** 2).sum(axis=-1))
    similarity = weights @ np.exp(-gamma * ((X - z) ** 2).sum(axis=1))
    expected = -similarity / (weights @ kernel @ weights)

    assert model.score_partial(z) == pytest.approx(expected, rel=0.02)


def test_decay_forgets_old_instances():
    from pysad.models import EXPoSE

    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(size=(200, 2)), 5.0 + rng.normal(size=(200, 2))])

    running = EXPoSE(random_state=0).fit(X)
    forgetting = EXPoSE(decay=0.05, random_state=0).fit(X)

    # The running mean still holds the first cluster, while gradual forgetting has moved on.
    assert running.score_partial(np.zeros(2)) < running.score_partial(np.full(2, 2.5))
    assert forgetting.score_partial(np.zeros(2)) > forgetting.score_partial(np.full(2, 5.0))


def test_outlier_scores_higher():
    from pysad.models import EXPoSE

    X = np.random.RandomState(0).normal(size=(500, 2))
    model = EXPoSE(random_state=0).fit(X)

    assert model.score_partial(np.array([4.0, 4.0])) > model.score_partial(np.zeros(2))


def test_score_is_zero_before_fitting():
    from pysad.models import EXPoSE

    assert EXPoSE().score_partial(np.zeros(3)) == 0.0


@pytest.mark.parametrize("decay", [-0.1, 1.0])
def test_rejects_decay_outside_the_unit_interval(decay):
    from pysad.models import EXPoSE

    with pytest.raises(ValueError, match="decay"):
        EXPoSE(decay=decay)
