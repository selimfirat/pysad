import numpy as np
import pytest


def _paper_score(window, q, z, kappa, m):
    """The score of the last value of `window`, from Eqs. (1) to (9) of the paper, one step at a time."""
    n = len(window)
    g = sum((window[-1] - window[-1 - i]) / i for i in range(1, m + 1)) / m
    x = np.concatenate([window, [window[n - m] + g * m] * kappa])

    F = np.fft.fft(x)
    L = np.log(np.abs(F))
    AL = np.array([L[max(0, f - q + 1) : f + 1].mean() for f in range(len(x))])
    S = np.abs(np.fft.ifft(np.exp(L - AL + 1j * np.angle(F))))
    S_bar = S[n - 1 - z : n - 1].mean()
    return (S[n - 1] - S_bar) / S_bar


def test_spectral_residual_scores_follow_the_equations_of_the_paper():
    from pysad.models import SpectralResidual

    X = np.random.default_rng(0).normal(size=40)
    params = {
        "window_size": 16,
        "filter_size": 3,
        "num_average_points": 6,
        "num_estimated_points": 4,
        "num_gradient_points": 3,
    }

    scores = SpectralResidual(**params).fit_score(X.reshape(-1, 1))

    expected = [_paper_score(X[i - 15 : i + 1], q=3, z=6, kappa=4, m=3) for i in range(15, 40)]
    np.testing.assert_allclose(scores[15:], expected, rtol=1e-10)


def test_spectral_residual_scores_match_the_saliency_map_of_the_authors_code():
    from pysad.models import SpectralResidual

    X = np.sin(2 * np.pi * np.arange(24) / 8)
    X[20] += 3.0

    scores = SpectralResidual(
        window_size=16,
        filter_size=3,
        num_average_points=6,
        num_estimated_points=4,
        num_gradient_points=3,
    ).fit_score(X.reshape(-1, 1))

    # Eq. (7) on the saliency map of spectral_residual_transform in https://github.com/microsoft/anomalydetector
    # (commit a3260ea), applied to the window that the model extends with Eqs. (8) and (9).
    expected = [
        -0.5284428184,
        -0.1477996531,
        -0.319879954,
        -0.9173249264,
        -0.5284428184,
        0.7544467932,
        0.06439015293,
        -0.2497613378,
        -0.6798316253,
    ]
    np.testing.assert_allclose(scores[15:], expected, rtol=1e-9)


def test_spectral_residual_warmup():
    from pysad.models import SpectralResidual

    X = np.random.default_rng(0).normal(size=(50, 1))

    scores = SpectralResidual(window_size=30).fit_score(X)

    assert np.all(scores[:29] == 0.0)
    assert np.all(scores[29:] != 0.0)


def test_spectral_residual_scores_do_not_depend_on_the_scale():
    from pysad.models import SpectralResidual

    X = np.random.default_rng(0).normal(size=(200, 1))

    scores = SpectralResidual().fit_score(X)

    np.testing.assert_allclose(
        SpectralResidual().fit_score(1000.0 * X), scores, rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize("value", [0.0, 5.0])
def test_spectral_residual_scores_a_constant_series_as_zero(value):
    from pysad.models import SpectralResidual

    scores = SpectralResidual(window_size=30).fit_score(np.full((60, 1), value))

    np.testing.assert_allclose(scores, 0.0, atol=1e-9)


def test_spectral_residual_rejects_multivariate_instances():
    from pysad.models import SpectralResidual

    with pytest.raises(ValueError, match="univariate"):
        SpectralResidual().fit_partial(np.zeros(2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"filter_size": 0},
        {"num_estimated_points": 0},
        {"window_size": 21},
        {"window_size": 10, "num_average_points": 5, "num_gradient_points": 10},
    ],
)
def test_spectral_residual_rejects_invalid_parameters(kwargs):
    from pysad.models import SpectralResidual

    with pytest.raises(ValueError):
        SpectralResidual(**kwargs)
