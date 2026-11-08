import numpy as np
import pytest

MEANS = np.array([[0.2, 0.2], [0.8, 0.7]])
STD = 0.05


def _mixture_stream(rng, num_instances):
    """An even mixture of two Gaussians with standard deviation STD around MEANS, shuffled."""
    labels = rng.integers(0, 2, num_instances)
    return MEANS[labels] + rng.normal(0.0, STD, (num_instances, 2))


@pytest.mark.parametrize("seed", range(3))
def test_smart_sifter_learns_a_stationary_mixture(seed):
    from pysad.models import SmartSifter

    rng = np.random.default_rng(seed)
    model = SmartSifter(num_components=2, discount=0.002, random_state=seed)
    model.fit(_mixture_stream(rng, 10_000))

    order = np.argsort(model.means_[:, 0])
    np.testing.assert_allclose(model.weights_[order], [0.5, 0.5], atol=0.05)
    np.testing.assert_allclose(model.means_[order], MEANS, atol=0.02)
    # The smoothing term alpha * r / k gives each component a small share of the other cluster.
    stds = np.sqrt(np.diagonal(model.covariances_[order], axis1=1, axis2=2))
    np.testing.assert_allclose(stds, STD, rtol=0.25)


def test_smart_sifter_scores_the_log_loss_under_the_mixture():
    from scipy.stats import multivariate_normal

    from pysad.models import SmartSifter

    rng = np.random.default_rng(0)
    X = _mixture_stream(rng, 500)
    model = SmartSifter(random_state=0).fit(X[:-1])

    density = sum(
        weight * multivariate_normal(mean, cov + 1e-6 * np.eye(2)).pdf(X[-1])
        for weight, mean, cov in zip(model.weights_, model.means_, model.covariances_, strict=True)
    )
    np.testing.assert_allclose(model.score_partial(X[-1]), -np.log(density))
    np.testing.assert_allclose(model.fit_score_partial(X[-1]), -np.log(density))


def _steps_to_follow_a_jump(discount):
    from pysad.models import SmartSifter

    rng = np.random.default_rng(0)
    model = SmartSifter(num_components=1, discount=discount, random_state=0)
    model.fit(rng.normal(0.2, STD, (3000, 2)))
    for step, x in enumerate(rng.normal(0.8, STD, (3000, 2)), start=1):
        model.fit_partial(x)
        if np.abs(model.means_ - 0.8).max() < 0.05:
            return step

    return None


def test_smart_sifter_follows_a_jump_faster_with_a_larger_discount():
    slow, fast = _steps_to_follow_a_jump(0.005), _steps_to_follow_a_jump(0.02)

    assert fast is not None and slow is not None
    # The weight of the old points decays as (1 - r)^t.
    assert 3 * fast < slow


@pytest.mark.parametrize("noise", [0.0, 1e-9])
def test_smart_sifter_stays_finite_on_constant_features(noise):
    from pysad.models import SmartSifter

    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=20_000), 3.0 + noise * rng.normal(size=20_000)])
    model = SmartSifter(random_state=0)
    scores = model.fit_score(X)

    assert np.isfinite(scores).all()
    # A point off the constant value is far less likely than a point on it.
    assert model.score_partial(np.array([0.0, 3.1])) > model.score_partial(np.array([0.0, 3.0]))


@pytest.mark.parametrize(
    "kwargs",
    [{"discount": 0.0}, {"discount": 1.0}, {"discount": 0.6, "alpha": 2.0}, {"alpha": -1.0}],
)
def test_smart_sifter_rejects_invalid_discounting(kwargs):
    from pysad.models import SmartSifter

    with pytest.raises(ValueError):
        SmartSifter(**kwargs)
