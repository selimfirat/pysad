from collections import Counter, defaultdict

import numpy as np


class _ReferenceChains:
    # A direct port of chains_add in the authors' code (cmuxstream-core, cpp/chain.cpp). The bin widths
    # and shifts are set once from the first window, as compute_deltamax in cpp/main.cpp does.

    def __init__(self, first_window, fs, rand_arr):
        self.fs = fs
        self.deltamax = (first_window.max(axis=0) - first_window.min(axis=0)) / 2.0
        self.deltamax[np.abs(self.deltamax) <= 0.0001] = 1.0
        self.shift = rand_arr * self.deltamax
        self.counts = defaultdict(Counter)

    def add(self, y, update):
        # Returns the score negated, so that higher is more anomalous.
        n_chains, depth = self.fs.shape
        total = 0.0
        for c in range(n_chains):
            prebin = np.zeros(len(y))
            used = np.zeros(len(y), dtype=bool)
            best = np.inf
            for d in range(depth):
                f = self.fs[c, d]
                if not used[f]:
                    prebin[f] = (y[f] + self.shift[c, f]) / self.deltamax[f]
                    used[f] = True
                else:
                    prebin[f] = 2.0 * prebin[f] - self.shift[c, f] / self.deltamax[f]
                key = tuple(np.floor(prebin).astype(int))
                if update:
                    self.counts[c, d][key] += 1
                best = min(best, np.log2(self.counts[c, d][key] + 1) + d + 1)
            total += best
        return -total / n_chains


def _model_and_projections(X, window_size):
    from pysad.models import xStream

    model = xStream(num_components=8, n_chains=6, depth=10, window_size=window_size, random_state=0)
    Y = np.vstack([model.streamhash.transform_partial(x) for x in X])
    return model, Y


def test_xstream_scores_with_the_bins_of_the_first_window():
    rng = np.random.RandomState(0)
    X = rng.randn(61, 6)
    Z = 2.0 * rng.randn(30, 6)
    model, Y = _model_and_projections(X, window_size=30)

    # After 61 instances, the reference is the second window.
    model.fit(X)
    reference = _ReferenceChains(Y[:30], model.hs_chains.fs, model.hs_chains.rand_arr)
    for y in Y[30:60]:
        reference.add(y, update=True)
    expected = [reference.add(model.streamhash.transform_partial(z), update=False) for z in Z]

    np.testing.assert_allclose(model.score(Z), expected, rtol=1e-12, atol=0)


def test_xstream_scores_each_instance_with_the_previous_window():
    # The windowed mode of cpp/main.cpp: each instance after the first window is scored with the counts
    # of the previous full window, and the counts are rebuilt once the current window is full.
    rng = np.random.RandomState(0)
    X = rng.randn(130, 6)
    X[::17] += 4.0
    model, Y = _model_and_projections(X, window_size=20)

    scores = model.fit_score(X)

    reference = _ReferenceChains(Y[:20], model.hs_chains.fs, model.hs_chains.rand_arr)
    for y in Y[:20]:
        reference.add(y, update=True)
    expected = []
    for start in range(20, 130, 20):
        window = Y[start : start + 20]
        expected += [reference.add(y, update=False) for y in window]
        reference.counts.clear()
        for y in window:
            reference.add(y, update=True)

    np.testing.assert_allclose(scores[20:], expected, rtol=1e-12, atol=0)


def test_xstream_fit_score_partial_matches_score_then_fit():
    from pysad.models import xStream

    X = np.random.RandomState(0).randn(65, 6)
    model = xStream(num_components=8, n_chains=6, depth=10, window_size=20, random_state=0)
    reference = xStream(num_components=8, n_chains=6, depth=10, window_size=20, random_state=0)

    for x in X:
        expected = reference.score_partial(x)
        reference.fit_partial(x)
        assert model.fit_score_partial(x) == expected

    assert model.hs_chains.counts == reference.hs_chains.counts
    assert model.hs_chains.counts_cur == reference.hs_chains.counts_cur


def test_xstream_score_explain_partial_returns_score_and_shap():
    import numpy as np

    from pysad.models import xStream

    model = xStream(num_components=10, n_chains=6, depth=5, window_size=20, random_state=0)
    X = np.random.RandomState(0).randn(60, 6)
    model.fit(X[:40])

    score, shap = model.score_explain_partial(X[41])

    assert isinstance(score, float)
    # SHAP values are in projected space (num_components dimensions)
    assert shap.shape == (10,)
    assert shap.dtype == np.float64


def test_xstream_score_explain_partial_does_not_change_state():
    import numpy as np

    from pysad.models import xStream

    model = xStream(num_components=10, n_chains=6, depth=5, window_size=20, random_state=0)
    X = np.random.RandomState(0).randn(60, 6)
    model.fit(X[:40])

    before = model.score_partial(X[41])
    score, _ = model.score_explain_partial(X[41])
    after = model.score_partial(X[41])

    assert before == after
    assert score == before


def test_xstream_score_explain_partial_is_reproducible():
    import numpy as np

    from pysad.models import xStream

    X = np.random.RandomState(0).randn(60, 6)

    model_a = xStream(num_components=10, n_chains=6, depth=5, window_size=20, random_state=0)
    model_a.fit(X[:40])
    score_a, shap_a = model_a.score_explain_partial(X[41])

    model_b = xStream(num_components=10, n_chains=6, depth=5, window_size=20, random_state=0)
    model_b.fit(X[:40])
    score_b, shap_b = model_b.score_explain_partial(X[41])

    assert score_a == score_b
    np.testing.assert_array_equal(shap_a, shap_b)


def test_xstream_score_explain_partial_shap_depends_on_instance():
    import numpy as np

    from pysad.models import xStream

    model = xStream(num_components=10, n_chains=6, depth=5, window_size=20, random_state=0)
    X = np.random.RandomState(0).randn(60, 6)
    model.fit(X[:40])

    normal_score, normal_shap = model.score_explain_partial(X[41])
    outside_score, outside_shap = model.score_explain_partial(np.full(6, 100.0))

    assert normal_score != outside_score
    assert not np.allclose(normal_shap, outside_shap)


def test_xstream_score_explain_partial_returns_zeros_before_first_window():
    import numpy as np

    from pysad.models import xStream

    model = xStream(num_components=10, n_chains=6, depth=5, window_size=20, random_state=0)
    # Before any instance is fitted, score_partial returns -1.0 (min depth)
    # but score_explain_partial returns zeros for SHAP values
    score, shap = model.score_explain_partial(np.random.RandomState(1).randn(6))

    assert score == -1.0  # score_partial returns -min(depth) when no counts
    assert shap.shape == (10,)  # num_components
    assert np.all(shap == 0.0)


def test_xstream_score_explain_partial_rejects_nan():
    import numpy as np
    import pytest

    from pysad.models import xStream

    model = xStream(num_components=10, n_chains=6, depth=5, window_size=20, random_state=0)
    X = np.random.RandomState(0).randn(60, 6)
    model.fit(X[:40])

    anomalous = X[41].copy()
    anomalous[1] = np.nan
    with pytest.raises(ValueError, match="xStream does not accept NaN values"):
        model.score_explain_partial(anomalous)
