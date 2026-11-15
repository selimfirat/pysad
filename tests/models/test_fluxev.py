import numpy as np
import pytest


@pytest.mark.parametrize(
    "period,num_periods,drift,expected",
    [
        (2, 2, 0, [0, 0, 0, 0, 0, 0, 3, 3]),
        (3, 2, 1, [0, 0, 0, 0, 0, 0, 2, 2]),
        (2, 3, 0, [0, 0, 0, 0, 0, 0, 3, 2]),
        (None, 7, 2, [0, 0, 0, 1, 0, 0, 3, 3]),
    ],
)
def test_two_step_smoothing(period, num_periods, drift, expected):
    from pysad.models import FluxEV

    model = FluxEV(
        window_size=1, period=period, num_periods=num_periods, drift=drift, calibration_size=100
    )
    # Equations (3)-(7) give F = [0, 1, 0, 0, 3, 3] from the third point.
    scores = model.fit_score(np.array([0, 0, 0, 2, 4, 6, 14, 28])[:, None])
    np.testing.assert_allclose(scores, expected)


@pytest.mark.parametrize("period,baseline", [(1, np.sqrt(2)), (None, 0)])
def test_weighted_prediction(period, baseline):
    from pysad.models import FluxEV

    model = FluxEV(
        window_size=2, period=period, num_periods=2, drift=0, calibration_size=100, alpha=0.5
    )
    model.fit(np.array([0, 0, 0, 0, 0, 3])[:, None])
    # Equation (1) predicts 2. Errors [0, 3, 8] have variance 98/9.
    expected = max(np.sqrt(98 / 9) - 1.5 - baseline, 0)
    assert model.score_partial(np.array([10])) == pytest.approx(expected)


@pytest.mark.parametrize("period,warmup", [(2, 4), (None, 2)])
def test_constant_calibration_and_alarm(period, warmup):
    from pysad.models import FluxEV

    model = FluxEV(window_size=1, period=period, num_periods=2, drift=0, calibration_size=10)
    model.fit(np.zeros((warmup + 9, 1)))
    assert model.threshold_ == np.inf
    model.fit_partial(np.array([0]))
    assert model.threshold_ == 0
    assert model.score_partial(np.array([10])) == 5
    model.fit_partial(np.array([10]))
    assert model.threshold_ == 0
    model.fit(np.array([[10], [10]]))
    # Earlier alarms cannot raise the periodic baseline for this candidate.
    assert model.score_partial(np.array([30])) == 10
    assert np.isfinite(model.fit_score(np.zeros((100, 1)))).all()


def test_mom_calibration():
    from pysad.models import FluxEV

    model = FluxEV(window_size=1, period=2, num_periods=2, drift=0, calibration_size=100)
    # For X_i = 2**i, Equations (1)-(7) give S_i = 3 * 2**(i-5).
    fluctuations = 3 * 2.0 ** np.arange(-1, 99)
    initial = np.quantile(fluctuations, 0.98)
    peaks = fluctuations[fluctuations > initial] - initial
    mean = peaks.mean()
    ratio = mean**2 / peaks.var(ddof=1)
    shape = (1 - ratio) / 2
    scale = mean * (1 + ratio) / 2
    expected = initial + scale / shape * ((100 * 0.001 / len(peaks)) ** (-shape) - 1)
    model.fit((2.0 ** np.arange(104))[:, None])
    assert model.threshold_ == pytest.approx(expected)

    # Choose the next raw value to produce a non-alarm peak between both thresholds.
    score = (initial + expected) / 2
    value = 2.0**103 + 2.0**102 + 2 * (2.0**99 + score)
    assert model.score_partial(np.array([value])) == pytest.approx(score)
    model.fit_partial(np.array([value]))
    peaks = np.append(peaks, score - initial)
    mean = peaks.mean()
    ratio = mean**2 / peaks.var(ddof=1)
    shape = (1 - ratio) / 2
    scale = mean * (1 + ratio) / 2
    expected = initial + scale / shape * ((101 * 0.001 / len(peaks)) ** (-shape) - 1)
    assert model.threshold_ == pytest.approx(expected)
