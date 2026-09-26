
def test_loda_projections_stay_sparse_and_fixed():
    from pysad.models import LODA
    import numpy as np
    from pysad.utils import fix_seed

    fix_seed(61)
    X = np.random.rand(500, 16)

    model = LODA(num_bins=10, num_random_cuts=50)
    model.fit_partial(X[0])
    initial_projections = model.projections_.copy()

    # sqrt(16) = 4 non-zero components per projection.
    assert np.all(np.count_nonzero(initial_projections, axis=1) == 4)

    for x in X[1:]:
        model.fit_partial(x)

    np.testing.assert_array_equal(model.projections_, initial_projections)
    assert np.all(np.count_nonzero(model.projections_, axis=1) == 4)


def test_loda_histograms_accumulate_counts():
    from pysad.models import LODA
    import numpy as np
    from pysad.utils import fix_seed

    fix_seed(61)
    X = np.random.randn(300, 5)

    model = LODA(num_bins=10, num_random_cuts=20)
    for i, x in enumerate(X):
        model.fit_partial(x)
        np.testing.assert_array_equal(model.histograms_.sum(axis=1), i + 1)

    # Every instance lies inside the range covered by the bins.
    projected = X.dot(model.projections_.T)
    assert np.all(projected >= model.bin_lows_)
    assert np.all(projected < model.bin_lows_ + model.n_bins * model.bin_widths_)

    # The histograms follow the data: binning all instances at once with the final bins gives the same counts,
    # except for values that lie on a bin edge up to floating point rounding.
    positions = (projected - model.bin_lows_) / model.bin_widths_
    on_edge = np.abs(positions - np.round(positions)) < 1e-9
    inds = np.floor(positions).astype(int)
    for i in range(model.n_random_cuts):
        expected = np.bincount(inds[:, i], minlength=model.n_bins)
        assert np.abs(model.histograms_[i] - expected).sum() <= 2 * on_edge[:, i].sum()


def test_loda_histograms_extend_to_new_ranges():
    from pysad.models import LODA
    import numpy as np
    from pysad.utils import fix_seed

    fix_seed(61)
    model = LODA(num_bins=4, num_random_cuts=5)
    for x in np.linspace(0., 1., 20):
        model.fit_partial(np.array([x]))
    for x in [-50., 1000.]:
        model.fit_partial(np.array([x]))

    np.testing.assert_array_equal(model.histograms_.sum(axis=1), 22)
    projected = np.array([-50., 1000.])[:, None] * model.projections_[:, 0]
    assert np.all(projected >= model.bin_lows_)
    assert np.all(projected < model.bin_lows_ + model.n_bins * model.bin_widths_)


def test_loda_scores_outliers_higher():
    from pysad.models import LODA
    import numpy as np
    from pysad.utils import fix_seed

    fix_seed(61)
    X = np.random.randn(1000, 5)
    model = LODA()
    for x in X:
        model.fit_partial(x)

    normal_score = model.score_partial(np.zeros(5))
    outlier_score = model.score_partial(np.full(5, 8.))
    assert normal_score.shape == (1,)
    assert outlier_score[0] > normal_score[0]


def test_loda_auroc_synthetic_stream():
    from pysad.models import LODA
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from pysad.utils import fix_seed

    fix_seed(61)
    X = np.random.randn(1000, 5)
    y = np.zeros(1000, dtype=int)
    outliers = np.random.choice(np.arange(100, 1000), 30, replace=False)
    X[outliers] = np.random.uniform(3, 5, size=(30, 5)) * np.random.choice([-1, 1], size=(30, 5))
    y[outliers] = 1

    model = LODA()
    scores = np.array([model.fit_score_partial(x) for x in X]).ravel()

    assert roc_auc_score(y, scores) > 0.95


def test_loda_auroc_arrhythmia(test_path):
    from pysad.models import LODA
    import numpy as np
    import os
    from sklearn.metrics import roc_auc_score
    from pysad.utils import Data, fix_seed

    fix_seed(61)
    data = Data(os.path.join(test_path, "../../examples/data"))
    X, y = data.get_data("arrhythmia.mat")

    model = LODA()
    scores = np.array([model.fit_score_partial(x) for x in X]).ravel()

    assert roc_auc_score(y, scores) > 0.7


def test_loda_score_before_fit_and_non_finite_values():
    from pysad.models import LODA
    import numpy as np
    from pysad.utils import fix_seed

    fix_seed(61)
    model = LODA(num_bins=5, num_random_cuts=10)
    assert model.score_partial(np.zeros(4)).shape == (1,)

    # Instances with non-finite values are skipped instead of stretching the bins without bound.
    model.fit_partial(np.array([np.nan, 0., 0., 0.]))
    assert model.num_seen_ == 0
    for x in np.random.randn(50, 4):
        model.fit_partial(x)
    model.fit_partial(np.array([np.inf, 0., 0., 0.]))
    assert np.all(np.isfinite(model.bin_lows_)) and np.all(np.isfinite(model.bin_widths_))
    assert model.num_seen_ == 50
    np.testing.assert_array_equal(model.histograms_.sum(axis=1), 50)
    assert np.isfinite(model.score_partial(np.array([np.inf, 0., 0., 0.]))[0])
