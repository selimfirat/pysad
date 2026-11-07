def generate_stream(seed=61):
    import numpy as np

    rng = np.random.RandomState(seed)
    X = rng.normal(0.0, 1.0, size=(1000, 2))
    y = np.zeros(X.shape[0], dtype=int)

    outlier_idx = rng.choice(np.arange(100, X.shape[0]), size=30, replace=False)
    X[outlier_idx] += rng.choice([-8.0, 8.0], size=(30, 2))
    y[outlier_idx] = 1

    return X, y


def test_kelos_auroc():
    from sklearn.metrics import roc_auc_score

    from pysad.models import KELOS

    X, y = generate_stream()

    scores = KELOS(window_size=500, radius=0.5, num_neighbors=50).fit_score(X)

    assert scores.shape == (X.shape[0],)
    assert roc_auc_score(y, scores) > 0.95


def _klome(window, x, k):
    """KLOME of Definition 3.1 with every window instance as a kernel center of weight 1."""
    import numpy as np

    def density(p):
        dists = np.linalg.norm(window - p, axis=1)
        nearest = np.argsort(dists)[:k]
        h = 1.06 * window[nearest].std(axis=0).dot(window[nearest].std(axis=0)) ** 0.5
        h *= k ** (-1 / (window.shape[1] + 1))
        return np.mean(
            np.exp(-(dists[nearest] ** 2) / (2 * h**2)) / (np.sqrt(2 * np.pi) * h)
        ), nearest

    f_x, nearest = density(x)
    f_centers = np.array([density(window[j])[0] for j in nearest])
    return (f_x - f_centers.mean()) / f_centers.std()


def test_kelos_scores_minus_klome_when_every_instance_is_a_cluster():
    """With radius 0, each instance of the window is an abstract kernel center of weight 1."""
    import numpy as np

    from pysad.models import KELOS

    X = np.random.RandomState(3).normal(size=(120, 3))
    window_size, k = 40, 6

    scores = KELOS(window_size=window_size, radius=0.0, num_neighbors=k).fit_score(X)

    for t in range(k + 1, len(X)):
        window = X[max(0, t - window_size + 1) : t + 1]  # the instance enters its window
        np.testing.assert_allclose(scores[t], -_klome(window, X[t], k), rtol=1e-6)


def test_kelos_ranks_a_point_between_clusters_above_their_members():
    import numpy as np

    from pysad.models import KELOS

    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(0.0, 0.05, size=(200, 2)), rng.normal(3.0, 0.5, size=(200, 2))])
    model = KELOS(window_size=400, radius=0.1, num_neighbors=10).fit(rng.permutation(X))

    members = model.score(np.array([[0.0, 0.0], [3.0, 3.0]]))
    assert model.score_partial(np.array([0.5, 0.5])) > members.max()


def test_kelos_scores_0_until_the_densities_differ():
    import numpy as np

    from pysad.models import KELOS

    model = KELOS(radius=1.0)
    assert model.score_partial(np.zeros(2)) == 0.0
    model.fit_partial(np.zeros(2))
    assert model.score_partial(np.full(2, 0.1)) == 0.0  # the window holds one cluster
