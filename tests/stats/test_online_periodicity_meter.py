import pickle

import numpy as np
import pytest

from pysad.statistics import OnlinePeriodicityMeter


def test_waits_for_a_full_window():
    meter = OnlinePeriodicityMeter()
    assert meter.get() == 0.0
    for value in [-1, 8, -2, 5, 1, -4]:
        assert meter.update(value) is meter
        assert meter.get() == 0.0


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([-1, 8, -2, 5, 1, -4], 3.0),
        ([0, 1, 0, -1, 0, 1, 0, -1], 4.0),
        ([0, 1, 0, 0, 0, -1, 0, 0], 8.0),
        ([0] * 8, 0.0),
        ([3] * 8, 0.0),
        ([1, 0, 0, 0, 0, 0, 0, 0], 0.0),
        ([1, -1] * 4, 0.0),
    ],
    ids=["review-input", "period-four", "equal-peaks", "zero", "constant", "dc-tie", "nyquist"],
)
def test_initial_period(values, expected):
    # The review input has powers 49/6, 73/6, and 175/6 at bins 0, 1, and 2.
    # Equal peaks at bins 1 and 3 select bin 1. A tie with DC selects DC.
    meter = OnlinePeriodicityMeter(window_size=len(values))
    for value in values:
        meter.update(value)
    assert meter.get() == expected
    assert isinstance(meter.get(), float)


@pytest.mark.parametrize("window_size", [9, 24])
def test_sliding_period_matches_batch_fourier_transform(window_size):
    # An independent batch transform detects errors in the recursive window update.
    times = np.arange(6 * window_size)
    values = np.sin(2 * np.pi * times / 4) + 0.6 * np.cos(2 * np.pi * times / 7)
    meter = OnlinePeriodicityMeter(window_size=window_size)
    for index, value in enumerate(values):
        meter.update(value)
        if index + 1 < window_size:
            assert meter.get() == 0.0
            continue
        window = values[index + 1 - window_size : index + 1]
        spectrum = np.fft.fft(window)[: window_size // 2]
        peak = int(np.argmax(abs(spectrum) ** 2))
        expected = window_size / peak if peak else 0.0
        assert meter.get() == expected


def test_replaces_an_old_period():
    meter = OnlinePeriodicityMeter(window_size=24)
    for value in [0, 1, 0, -1] * 12:
        meter.update(value)
    assert meter.get() == 4.0
    for value in [0, 1, 1, 0, -1, -1] * 4:
        meter.update(value)
    assert meter.get() == 6.0


@pytest.mark.parametrize("prefix_length", [3, 12])
def test_pickle_preserves_future_periods(prefix_length):
    meter = OnlinePeriodicityMeter(window_size=12)
    values = [0, 1, 0, -1] * 12
    for value in values[:prefix_length]:
        meter.update(value)
    restored = pickle.loads(pickle.dumps(meter))
    assert restored.get() == meter.get()
    for value in values[prefix_length:]:
        meter.update(value)
        restored.update(value)
        assert restored.get() == meter.get()
    assert restored.get() == 4.0


@pytest.mark.parametrize("window_size", [0, 1, 2, 3])
def test_rejects_windows_without_a_supported_frequency(window_size):
    with pytest.raises(ValueError, match="window_size"):
        OnlinePeriodicityMeter(window_size=window_size)


def test_remove_raises():
    meter = OnlinePeriodicityMeter()
    meter.update(1.0)
    with pytest.raises(NotImplementedError):
        meter.remove(1.0)
