import gc
import weakref

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from pysad.models import Inqmad
from pysad.models.inqmad import _adapt_features

PARAMS = {"input_shape": 3, "dim_x": 32, "initial_size": 20, "random_state": 0}


def _model(**kwargs):
    return Inqmad(**{**PARAMS, **kwargs})


def _states(model, X):
    """The states of the rows of X, min-max scaled with the initial sample."""
    X = (np.atleast_2d(X) - model.feature_min) / model.feature_range
    psi = np.cos(X @ model.weights + model.offset)
    return psi / np.linalg.norm(psi, axis=1, keepdims=True)


def test_initial_sample_sets_rho_to_its_mean_state_and_tau_to_its_beta_quantile():
    """Paper Eq. 2 and Algorithm 1, step 6: once the initial sample is
    complete, it sets the min-max scaling, rho is the mean of its scaled
    psi psi^T and tau is the beta quantile of its densities psi^T rho psi.
    """
    X = np.random.default_rng(0).random((20, 3)) * [1.0, 10.0, 100.0] - 3.0
    model = _model(beta=0.2)
    model.fit(X[:19])
    assert model.rho is None

    model.fit_partial(X[19])

    np.testing.assert_array_equal(model.feature_min, X.min(axis=0))
    np.testing.assert_array_equal(model.feature_range, np.ptp(X, axis=0))
    psi = _states(model, X)
    rho = psi.T @ psi / 20
    np.testing.assert_allclose(model.rho, rho, atol=1e-12)
    assert model.rho.dtype == np.float64
    assert model.tau == pytest.approx(np.quantile(np.einsum("ni,ij,nj->n", psi, rho, psi), 0.2))


def test_normal_instance_updates_rho_with_forgetting():
    """Paper Eq. 4 and Algorithm 2: an instance whose density is at least
    tau updates rho to (1 - alpha) rho + alpha psi psi^T.
    """
    rng = np.random.default_rng(0)
    model = _model(alpha=0.3).fit(rng.random((20, 3)))
    x = np.array([0.5, 0.5, 0.5])
    psi = _states(model, x)[0]
    rho = model.rho.copy()
    assert psi @ rho @ psi >= model.tau

    model.fit_partial(x)

    np.testing.assert_allclose(model.rho, 0.7 * rho + 0.3 * np.outer(psi, psi), atol=1e-12)


def test_anomalous_instance_leaves_rho_unchanged():
    """Paper Algorithm 2, step 6: an instance whose density is below tau
    is classified as an anomaly and does not change rho.
    """
    rng = np.random.default_rng(0)
    model = _model().fit(rng.random((20, 3)))
    far_point = np.array([5.0, 5.0, 5.0])
    psi = _states(model, far_point)[0]
    rho = model.rho.copy()
    assert psi @ rho @ psi < model.tau

    model.fit_partial(far_point)

    np.testing.assert_array_equal(model.rho, rho)


def test_score_partial_is_negated_paper_density():
    """Paper Eq. 3: after the initial sample, the score is the negated
    density psi^T rho psi.
    """
    rng = np.random.default_rng(0)
    model = _model().fit(rng.random((30, 3)))
    q = rng.random(3)
    psi = _states(model, q)[0]

    assert model.score_partial(q) == pytest.approx(-psi @ model.rho @ psi, rel=1e-12)


def test_score_during_initial_sample_is_mean_squared_kernel():
    """While the initial sample fills up, an instance scores the negated
    mean of k(x, x_i)^2 over the sample so far, the density that
    psi^T rho psi approximates with rho the mean state of the sample, on
    instances min-max scaled with the sample so far. A feature with no
    range is not scaled.
    """
    rng = np.random.default_rng(0)
    X = rng.random((5, 3)) * [2.0, 3.0, 0.0]
    q = rng.random(3)
    model = _model(sigma=0.5).fit(X)

    scaled_range = np.ptp(X, axis=0) + [0.0, 0.0, 1.0]
    sq_dists = np.sum(((X - q) / scaled_range) ** 2, axis=1)
    expected = -np.mean(np.exp(-sq_dists / (2 * 0.5**2)) ** 2)

    assert model.score_partial(q) == pytest.approx(expected, rel=1e-12)


def test_defaults_follow_the_paper():
    """The paper uses D = 2000 features and gamma = 1 / (2 sigma^2)."""
    model = Inqmad(input_shape=3)

    assert model.weights.shape == (3, 2000)
    assert model.gamma == 0.5
    assert (model.initial_size, model.alpha, model.beta, model.learning_rate) == (
        256,
        0.01,
        0.1,
        1e-3,
    )


def test_scores_do_not_depend_on_the_scale_of_the_features():
    """The instances are min-max scaled with the initial sample, so an
    affine change of each feature leaves every score unchanged, except for
    the second instance, scored against a single instance with no range.
    """
    rng = np.random.default_rng(0)
    X = rng.random((60, 3))
    X[[25, 45]] += 2.0

    scores = _model().fit_score(X)
    rescaled = _model().fit_score(X * [10.0, 0.01, 3.0] + [5.0, -1.0, 0.0])

    np.testing.assert_allclose(np.delete(rescaled, 1), np.delete(scores, 1), atol=1e-9)


@pytest.mark.parametrize("name", ["sigma", "learning_rate"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_rejects_bandwidth_and_learning_rate_that_are_not_positive(name, value):
    with pytest.raises(ValueError, match=f"{name} must be"):
        _model(**{name: value})


def test_first_instance_scores_zero():
    """Before anything has been fitted the density is 0, so the score is
    0.0, the most anomalous possible score, instead of an error.
    """
    x = np.array([0.5, 0.5, 0.5])

    score = _model().score_partial(x)
    assert type(score) is float and score == 0.0

    score = _model().fit_score_partial(x)
    assert type(score) is float and score == 0.0


def test_fit_score_matches_score_then_fit():
    """Paper Algorithm 2: fit_score_partial scores each instance against
    rho_t, and then fits it, through the initial sample and after it.
    """
    rng = np.random.default_rng(0)
    X = rng.random((50, 3))
    X[[10, 30]] += 5.0

    fit_scores = _model().fit_score(X)

    model = _model()
    expected = []
    for x in X:
        expected.append(model.score_partial(x))
        model.fit_partial(x)

    np.testing.assert_array_equal(fit_scores, expected)


def test_fit_score_ranks_interleaved_far_outliers_above_inliers():
    """Regression test for #120: scores are anomaly scores (higher is more
    anomalous), and each depends on the data seen so far.
    """
    rng = np.random.default_rng(0)
    X = rng.random((300, 3))
    y = np.zeros(300)
    outliers = rng.choice(300, size=30, replace=False)
    X[outliers] += 5.0
    y[outliers] = 1

    scores = _model().fit_score(X)

    assert roc_auc_score(y, scores) > 0.9


def test_score_follows_later_training():
    """Regression test for #120: scoring after more fitting reflects the
    updated model, here normal instances next to the scored one.
    """
    rng = np.random.default_rng(0)
    q = np.array([1.5, 1.5, 1.5])
    model = _model(alpha=0.5).fit(rng.random((20, 3)) * 3)
    before = model.score_partial(q)

    model.fit(np.tile(q + 0.1, (5, 1)))

    assert model.score_partial(q) < before


def test_docstring_keeps_math_backslashes():
    """Regression test for #120: LaTeX such as \\rho and \\tau must not
    turn into carriage returns or tabs in the rendered docstring.
    """
    assert "\r" not in Inqmad.__doc__
    assert "\t" not in Inqmad.__doc__


def test_scores_use_the_current_random_features():
    """Regression test for #171: the Fourier features are read on every
    call, so replacing them after the model has scored takes effect.
    """
    rng = np.random.default_rng(0)
    X = rng.random((30, 3))
    q = np.array([0.5, 0.5, 0.5])
    model = _model().fit(X)
    other = _model(random_state=1).fit(X)
    model.score_partial(q)

    model.weights, model.offset, model.rho = other.weights, other.offset, other.rho

    assert model.score_partial(q) == other.score_partial(q)


def test_deleted_model_is_garbage_collected():
    """Regression test for #173: nothing at class or module level keeps a
    model or its random features alive after its last reference is gone.
    """
    rng = np.random.default_rng(0)
    model = _model().fit(rng.random((30, 3)))
    model.score_partial(rng.random(3))
    refs = [weakref.ref(model), weakref.ref(model.weights), weakref.ref(model.rho)]

    del model
    gc.collect()

    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("name", ["input_shape", "dim_x", "initial_size"])
@pytest.mark.parametrize("value", [0, -1])
def test_rejects_sizes_below_one(name, value):
    """Regression test for #172: a zero or negative size raises a
    ValueError that names the parameter.
    """
    with pytest.raises(ValueError, match=f"{name} must be at least 1, got {value}"):
        _model(**{name: value})


@pytest.mark.parametrize("name", ["input_shape", "dim_x", "initial_size"])
@pytest.mark.parametrize("value", [True, 3.0, "3", None])
def test_rejects_sizes_that_are_not_ints(name, value):
    """Regression test for #172: sizes must be ints, and a bool is
    rejected too.
    """
    with pytest.raises(TypeError, match=f"{name} must be an int"):
        _model(**{name: value})


@pytest.mark.parametrize("name", ["alpha", "beta"])
@pytest.mark.parametrize("value", [-0.1, 1.5, np.nan])
def test_rejects_rates_outside_the_unit_interval(name, value):
    with pytest.raises(ValueError, match=name):
        _model(**{name: value})


def test_accepts_numpy_int_sizes():
    """Regression test for #172: NumPy integers, such as a shape read off
    an array, are valid sizes.
    """
    model = _model(input_shape=np.int64(3), dim_x=np.int32(8), initial_size=np.int64(5))

    assert model.weights.shape == (3, 8)
    assert model.initial_size == 5


@pytest.mark.parametrize(
    "X",
    [
        np.zeros(2),
        np.zeros(4),
        np.zeros((1, 4)),
        np.float64(1.0),
        np.zeros((2, 3)),
        np.zeros((1, 1, 3)),
        [[]],
    ],
    ids=["too-few", "too-many", "row-too-many", "0-d", "two-rows", "3-d", "empty-row"],
)
@pytest.mark.parametrize("method", ["fit_partial", "score_partial", "fit_score_partial"])
@pytest.mark.parametrize("num_fitted", [5, 30])
def test_rejects_instances_of_the_wrong_shape(method, X, num_fitted):
    """Regression test for #172: an instance that is not one row of
    input_shape features raises a ValueError naming the expected shape and
    leaves the model unchanged, during the initial sample and after it.
    """
    rng = np.random.default_rng(0)
    model = _model().fit(rng.random((num_fitted, 3)))
    state = [None if a is None else a.copy() for a in (model.initial, model.rho)]

    with pytest.raises(ValueError, match=r"one instance of shape \(3,\)"):
        getattr(model, method)(X)

    for before, after in zip(state, (model.initial, model.rho), strict=True):
        np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize("method", ["fit_partial", "score_partial", "fit_score_partial"])
def test_accepts_one_row_instances(method):
    """Regression test for #172: a (1, input_shape) row and a list are the
    same instance as the (input_shape,) array.
    """
    rng = np.random.default_rng(0)
    X = rng.random((30, 3))
    x = rng.random(3)

    results = []
    for instance in [x, x[None, :], list(x)]:
        model = _model().fit(X)
        result = getattr(model, method)(instance)
        results.append((result if method != "fit_partial" else None, model.rho.copy()))

    for score, rho in results[1:]:
        assert score == results[0][0]
        np.testing.assert_array_equal(rho, results[0][1])


def _kernel_mse(X, Y, weights, offset, gamma):
    approx = np.sum(_rff_states(X, weights, offset) * _rff_states(Y, weights, offset), axis=1)
    return np.mean((np.exp(-gamma * np.sum((X - Y) ** 2, axis=1)) - approx) ** 2)


def _rff_states(X, weights, offset):
    psi = np.cos(X @ weights + offset)
    return psi / np.linalg.norm(psi, axis=1, keepdims=True)


def test_adaptive_features_step_follows_the_kernel_error_gradient():
    """Paper Section 3.1: the features are fitted by gradient descent on the
    squared error between the kernel and the inner product of the states.
    Adam's first step moves every parameter by the learning rate against
    the sign of its gradient, which a finite difference checks here.
    """
    rng = np.random.default_rng(0)
    X = rng.random((40, 3))
    model = _model(dim_x=8)
    weights, offset = model.weights, model.offset
    x, y = X[np.random.RandomState(1).randint(40, size=(2, 16))]

    new_weights, new_offset = _adapt_features(
        X, weights, offset, model.gamma, 1e-4, np.random.RandomState(1), num_steps=1, batch_size=16
    )

    for param, new in ((weights, new_weights), (offset, new_offset)):
        numeric = np.zeros_like(param)
        for i in np.ndindex(param.shape):
            delta = np.zeros_like(param)
            delta[i] = 1e-6
            w, o = (weights + delta, offset) if param is weights else (weights, offset + delta)
            w2, o2 = (weights - delta, offset) if param is weights else (weights, offset - delta)
            numeric[i] = (
                _kernel_mse(x, y, w, o, model.gamma) - _kernel_mse(x, y, w2, o2, model.gamma)
            ) / 2e-6
        np.testing.assert_allclose(new - param, -1e-4 * np.sign(numeric), rtol=1e-3)


def test_adaptive_features_lower_the_kernel_error_on_the_initial_sample():
    """Completing the initial sample fits the features, which then
    approximate the kernel on the scaled sample better than the random
    features they start from.
    """
    X = np.random.default_rng(0).random((20, 3))
    model = _model(dim_x=16, learning_rate=1e-2)
    weights, offset = model.weights.copy(), model.offset.copy()
    i, j = np.triu_indices(20, k=1)

    model.fit(X)

    scaled = (X - model.feature_min) / model.feature_range
    before = _kernel_mse(scaled[i], scaled[j], weights, offset, model.gamma)
    after = _kernel_mse(scaled[i], scaled[j], model.weights, model.offset, model.gamma)
    assert after < 0.9 * before


def test_long_normal_stream_matches_step_by_step_rho_recurrence():
    """Fitting more normal instances than dim_x keeps every prequential
    score and the final density matrix equal to the exact Eq. 4 recurrence.
    """
    rng = np.random.default_rng(0)
    X_init = rng.random((4, 3))
    X_stream = X_init[rng.integers(4, size=20)] + rng.normal(0.0, 0.01, size=(20, 3))
    model = _model(dim_x=8, initial_size=4, alpha=0.2, beta=0.0).fit(X_init)

    psi_init = _states(model, X_init)
    rho = psi_init.T @ psi_init / len(psi_init)
    expected_scores = []
    num_updates = 0
    for x in X_stream:
        psi = _states(model, x)[0]
        density = float(psi @ rho @ psi)
        expected_scores.append(-density)
        if density >= model.tau:
            rho = 0.8 * rho + 0.2 * np.outer(psi, psi)
            num_updates += 1

    assert num_updates > 8
    scores = model.fit_score(X_stream)

    np.testing.assert_allclose(scores, expected_scores, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model.rho, rho, rtol=1e-12, atol=1e-12)
