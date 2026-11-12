from __future__ import annotations

import numpy as np

from pysad.core.base_statistic import BaseStatistic


class OnlinePeriodicityMeter(BaseStatistic):
    """Estimate a stream period with the original OnlineSLE algorithm.

    This meter follows Section 4.1, Algorithms 1 and 2, and Equation (3) of
    :cite:`phungtuaeng2026online`. Each input represents one equally spaced sample.
    Remove trends before you update the meter. The meter does not remove the mean.

    The first full window initializes the Fourier coefficients with an FFT.
    Each later update replaces the oldest sample and updates the coefficients with
    the sliding DFT recurrence. The meter keeps no EWMA mean or second moment.
    Memory and update cost depend on ``window_size``, not stream length.
    Initialization costs O(N log N). Later updates cost O(N), where N is ``window_size``.

    Selection follows the paper's bin rule, without spectral peak interpolation.
    The meter selects the largest raw power among bins 0 through ``N // 2 - 1``.
    This range follows the `reference implementation
    <https://github.com/thanapol2/onlineSLE_IJSCAI/blob/main/src/utilities.py>`_
    and excludes the Nyquist bin.
    An exact tie selects the smallest bin index. The period is ``N / bin_index``.
    Frequency resolution limits the estimate. Noise can also produce a period.

    Unlike the paper's initial batch input, this API collects the first window through
    ``update``. It returns 0.0 until that window fills. It also returns 0.0 when bin 0
    wins, because zero frequency has no finite period. No values undergo clipping.
    The paper leaves these edge cases undefined. Unlike its reference code, this
    meter keeps fractional periods and accepts bin 1, as the paper's formula specifies.

    Args:
        window_size (int): Number of samples in the sliding window.
            Use at least 4 samples (Default=100).

    Raises:
        ValueError: If ``window_size`` is less than 4.
    """

    def __init__(self, window_size: int = 100) -> None:
        if window_size < 4:
            raise ValueError("window_size must be at least 4.")

        self.window_size = window_size
        self.buffer_ = np.zeros(window_size, dtype=np.float64)
        self.position_ = 0
        self.ready_ = False
        self.spectrum_ = np.zeros(window_size // 2, dtype=np.complex128)
        self.twiddle_ = np.exp(2j * np.pi * np.arange(window_size // 2) / window_size)
        self.power_ = np.zeros(window_size // 2, dtype=np.float64)
        self.period_ = 0.0

    def update(self, num: float) -> OnlinePeriodicityMeter:
        """Add one sample and update the estimated period.

        Args:
            num (float): Next sample at an equally spaced time step.

        Returns:
            OnlinePeriodicityMeter: This meter.
        """
        oldest = self.buffer_[self.position_]
        self.buffer_[self.position_] = num
        self.position_ = (self.position_ + 1) % self.window_size

        if not self.ready_:
            if self.position_ != 0:
                return self
            self.spectrum_[:] = np.fft.fft(self.buffer_)[: self.window_size // 2]
            self.ready_ = True
        else:
            # Equation (3) shifts the window by one sample for every frequency bin.
            self.spectrum_ += num - oldest
            self.spectrum_ *= self.twiddle_

        np.abs(self.spectrum_, out=self.power_)
        self.power_ **= 2
        self.power_ /= self.window_size
        peak = int(self.power_.argmax())
        self.period_ = self.window_size / peak if peak else 0.0
        return self

    def get(self) -> float:
        """Return the period in samples.

        Returns:
            float: Estimated period, or 0.0 before a full window or when zero frequency wins.
        """
        return self.period_
