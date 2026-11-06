import numpy as np
import pytest


def _reference_scores(X, nu, learning_rate, gamma):
    """NORMA novelty detection without truncation: Eq. (20) of the paper, with the gradient signs for rho."""
    support_vectors, alphas, rho, scores = [], [], 0.0, []
    for x in X:
        f = sum(
            a * np.exp(-gamma * ((s - x) ** 2).sum())
            for s, a in zip(support_vectors, alphas, strict=True)
        )
        scores.append(rho - f)
        alphas = [(1.0 - learning_rate) * a for a in alphas]
        if f < rho:
            support_vectors.append(x)
            alphas.append(learning_rate)
            rho -= learning_rate * (1.0 - nu)
        else:
            rho += learning_rate * nu
    return np.array(scores)


def test_scores_equal_the_untruncated_update_while_the_window_is_not_full():
    from pysad.models import OnlineOneClassSVM

    X = np.random.RandomState(0).normal(size=(200, 3))
    model = OnlineOneClassSVM(nu=0.3, learning_rate=0.05, gamma=0.5, window_size=200)

    np.testing.assert_allclose(model.fit_score(X), _reference_scores(X, 0.3, 0.05, 0.5))


def test_truncation_drops_only_negligible_terms():
    from pysad.models import OnlineOneClassSVM

    X = np.random.RandomState(0).normal(size=(1000, 3))
    model = OnlineOneClassSVM(nu=0.3, learning_rate=0.05, gamma=0.5, window_size=300)

    # The dropped terms are below 0.05 * 0.95 ** 300 = 1e-8 each.
    np.testing.assert_allclose(
        model.fit_score(X), _reference_scores(X, 0.3, 0.05, 0.5), rtol=0.0, atol=1e-6
    )


def test_fraction_of_margin_errors_is_nu():
    from pysad.models import OnlineOneClassSVM

    X = np.random.RandomState(0).normal(size=(2000, 2))
    model = OnlineOneClassSVM(nu=0.2, window_size=2000).fit(X)

    assert np.count_nonzero(model.alphas_) / len(X) == pytest.approx(0.2, abs=0.01)


def test_outlier_scores_higher():
    from pysad.models import OnlineOneClassSVM

    X = np.random.RandomState(0).normal(size=(500, 2))
    model = OnlineOneClassSVM().fit(X)

    assert model.score_partial(np.array([6.0, 6.0])) > model.score_partial(np.zeros(2))


@pytest.mark.parametrize("params", [{"nu": 0.0}, {"nu": 1.5}, {"learning_rate": 1.0}])
def test_rejects_out_of_range_parameters(params):
    from pysad.models import OnlineOneClassSVM

    with pytest.raises(ValueError, match="nu must be in"):
        OnlineOneClassSVM(**params)
