import numpy as np
import pytest


def _series(n, period, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.002 * t + np.where(t >= n // 2, 1.0, 0.0)
    seasonal = np.sin(2 * np.pi * t / period) + 0.5 * np.cos(4 * np.pi * t / period)
    noise = 0.1 * rng.normal(size=n)
    return trend + seasonal + noise, noise


def _dense_residuals(y, v, period, lam, num_iterations):
    # Solves Eq. 7 of the paper on all points since the initialization at every step, with the
    # IRLS weights of Eq. 4 and 5, and returns the residual of the newest point of each step.
    v = v.copy()
    u, residuals = [], []
    p = [[] for _ in range(num_iterations)]
    q = [[] for _ in range(num_iterations)]
    for m in range(1, len(y) + 1):
        u.append(v[(m - 1) % period])
        A = np.zeros((2 * m, 2 * m))
        rhs = np.zeros(2 * m)
        for j in range(m):
            A[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] += [[1.0, 1.0], [1.0, 2.0]]
            rhs[2 * j : 2 * j + 2] = y[j], y[j] + u[j]
        p_new = q_new = 1.0
        for i in range(num_iterations):
            p[i].append(p_new)
            q[i].append(q_new)
            A_i = A.copy()
            for j in range(1, m):
                d = np.zeros(2 * m)
                d[[2 * j - 2, 2 * j]] = -1.0, 1.0
                A_i += lam * p[i][j] * np.outer(d, d)
            for j in range(2, m):
                d = np.zeros(2 * m)
                d[[2 * j - 4, 2 * j - 2, 2 * j]] = 1.0, -2.0, 1.0
                A_i += lam * q[i][j] * np.outer(d, d)
            x = np.linalg.solve(A_i, rhs)
            tau = x[::2]
            if m >= 2:
                p_new = 0.5 / (abs(tau[-1] - tau[-2]) + 1e-5)
            if m >= 3:
                q_new = 0.5 / (abs(tau[-1] - 2 * tau[-2] + tau[-3]) + 1e-5)
        residuals.append(y[m - 1] - x[-2] - x[-1])
        v[(m - 1) % period] = x[-1]
    return np.array(residuals)


@pytest.mark.parametrize("num_iterations", [1, 3])
def test_one_shot_stl_matches_the_exact_solution_of_the_paper_problem(num_iterations):
    from pysad.transform.preprocessing import OneShotSTL

    period, lam = 5, 2.0
    y, _ = _series(60, period, seed=1)
    num_init = 3 * period
    transformer = OneShotSTL(
        period=period, lam=lam, num_init_periods=3, num_iterations=num_iterations
    )
    transformer.fit(y[:num_init, None])
    v = transformer.seasonal_.copy()

    residuals = transformer.fit_transform(y[num_init:, None]).ravel()

    expected = _dense_residuals(y[num_init:], v, period, lam, num_iterations)
    np.testing.assert_allclose(residuals, expected, rtol=0, atol=1e-7)


def test_one_shot_stl_recovers_the_noise_of_a_series_with_trend_and_cycle():
    from pysad.transform.preprocessing import OneShotSTL

    period = 24
    y, noise = _series(2000, period)

    residuals = OneShotSTL(period=period).fit_transform(y[:, None]).ravel()

    assert np.all(residuals[: 4 * period] == 0.0)
    # The residual follows the noise, including after the level jumps at the middle.
    errors = np.abs(residuals - noise)[4 * period :]
    assert errors.mean() < 0.08
    assert np.abs(residuals - noise)[1050:].mean() < 0.08


def test_one_shot_stl_shift_window_follows_a_shifted_cycle():
    from pysad.transform.preprocessing import OneShotSTL

    period = 50
    t = np.arange(3000)
    noise = 0.05 * np.random.default_rng(0).normal(size=t.size)
    shifted = t + np.where(t > 1500, 5, 0)
    y = np.where(shifted % period < period // 2, 1.0, -1.0) + noise

    errors = {}
    for shift_window in (0, 10):
        transformer = OneShotSTL(period=period, shift_window=shift_window)
        errors[shift_window] = np.abs(transformer.fit_transform(y[:, None]).ravel() - noise)[1500:]

    assert errors[10].mean() < 0.7 * errors[0].mean()


def test_one_shot_stl_transform_partial_does_not_update():
    from pysad.transform.preprocessing import OneShotSTL

    period = 4
    y, _ = _series(40, period)
    transformer = OneShotSTL(period=period, shift_window=1)
    transformer.fit(y[:30, None])

    first = transformer.transform_partial(y[30:31])
    second = transformer.transform_partial(y[30:31])
    fitted = transformer.fit_transform_partial(y[30:31])

    assert first.shape == (1,)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, fitted)


def test_one_shot_stl_rejects_multivariate_input():
    from pysad.transform.preprocessing import OneShotSTL

    with pytest.raises(ValueError, match="univariate"):
        OneShotSTL(period=4).fit_partial(np.array([1.0, 2.0]))


@pytest.mark.parametrize(("period", "num_init_periods"), [(1, 4), (4, 2)])
def test_one_shot_stl_rejects_invalid_parameters(period, num_init_periods):
    from pysad.transform.preprocessing import OneShotSTL

    with pytest.raises(ValueError):
        OneShotSTL(period=period, num_init_periods=num_init_periods)
