from pysad.statistics.abs_statistic import AbsStatistic
from pysad.statistics.running_statistic import RunningStatistic


def test_all_zero_stats():
    import numpy as np

    from pysad.statistics import (
        AbsStatistic,
        AverageMeter,
        CountMeter,
        MaxMeter,
        MedianMeter,
        MinMeter,
        RunningStatistic,
        SumMeter,
        SumSquaresMeter,
        VarianceMeter,
    )
    from pysad.utils import fix_seed

    fix_seed(61)

    num_items = 100
    stat_classes = {
        AverageMeter: 0.0,
        CountMeter: "count",
        MaxMeter: 0.0,
        MedianMeter: 0.0,
        MinMeter: 0.0,
        SumMeter: 0.0,
        SumSquaresMeter: 0.0,
        VarianceMeter: 0.0,
    }

    for stat_cls, val in stat_classes.items():
        stat = stat_cls()
        abs_stat = AbsStatistic(stat_cls)
        window_size = 25
        running_stat = RunningStatistic(stat_cls, window_size=window_size)
        arr = np.zeros(num_items, dtype=np.float64)
        prev_value = 0.0
        for i in range(arr.shape[0]):
            num = arr[i]
            stat.update(num)
            abs_stat.update(num)
            running_stat.update(num)
            if i > 1:  # for variance meter.
                assert np.isclose(stat.get(), val if val != "count" else i + 1)
                assert np.isclose(abs_stat.get(), val if val != "count" else i + 1)
                assert np.isclose(
                    running_stat.get(), val if val != "count" else min(i + 1, window_size)
                )

                stat.remove(num)
                abs_stat.remove(num)
                assert np.isclose(stat.get(), prev_value)
                assert np.isclose(abs_stat.get(), abs(prev_value))
                stat.update(num)
                abs_stat.update(num)

            prev_value = stat.get()


def test_stats_with_batch_numpy():

    import numpy as np

    from pysad.statistics import (
        AverageMeter,
        CountMeter,
        MaxMeter,
        MedianMeter,
        MinMeter,
        SumMeter,
        SumSquaresMeter,
        VarianceMeter,
    )
    from pysad.utils import fix_seed

    fix_seed(61)

    num_items = 100
    stat_classes = {
        AverageMeter: np.mean,
        CountMeter: len,
        MaxMeter: np.max,
        MedianMeter: np.median,
        MinMeter: np.min,
        SumMeter: np.sum,
        SumSquaresMeter: lambda x: np.sum(x**2),
        VarianceMeter: np.var,
    }

    for stat_cls, val in stat_classes.items():
        stat = stat_cls()
        abs_stat = AbsStatistic(stat_cls)
        window_size = 25
        running_stat = RunningStatistic(stat_cls, window_size=window_size)

        arr = np.random.rand(num_items)
        prev_value = 0.0
        for i in range(arr.shape[0]):
            num = arr[i]
            stat.update(num)
            abs_stat.update(num)
            running_stat.update(num)

            if i > 1:  # for variance meter.
                assert np.isclose(stat.get(), val(arr[: i + 1]))
                assert np.isclose(running_stat.get(), val(arr[max(0, i - window_size + 1) : i + 1]))
                assert np.isclose(abs(stat.get()), abs_stat.get())

            stat.remove(num)
            abs_stat.remove(num)

            if i > 1:
                assert np.isclose(stat.get(), prev_value)
                assert np.isclose(abs_stat.get(), abs(prev_value))

            stat.update(num)
            abs_stat.update(num)

            prev_value = stat.get()


def test_running_statistic_passes_kwargs():
    import numpy as np

    from pysad.statistics import AverageMeter

    class ScaledAverage(AverageMeter):
        def __init__(self, scale=1.0):
            super().__init__()
            self.scale = scale

        def get(self):
            return self.scale * super().get()

    stat = RunningStatistic(ScaledAverage, window_size=2, scale=3.0)
    assert stat.statistic.scale == 3.0

    for num in [1.0, 2.0, 4.0]:
        stat.update(num)

    assert np.isclose(stat.get(), 3.0 * (2.0 + 4.0) / 2)


def test_variance_meter_is_zero_on_a_constant_stream():
    """#224: the sum-of-squares formula went negative here, from three instances of 0.1."""
    from pysad.statistics import VarianceMeter

    for value in [0.1, 316551.4375]:
        stat = VarianceMeter()
        for _ in range(100):
            assert stat.update(value).get() == 0.0


def test_variance_meter_keeps_precision_on_large_values():
    import numpy as np

    from pysad.statistics import VarianceMeter

    arr = 1e9 + np.arange(10, dtype=np.float64)
    stat = VarianceMeter()
    for num in arr:
        stat.update(num)

    assert np.isclose(stat.get(), np.var(arr), rtol=1e-12)


def test_running_variance_is_not_negative_when_the_window_turns_constant():
    import numpy as np

    from pysad.statistics import VarianceMeter

    window_size = 5
    running_stat = RunningStatistic(VarianceMeter, window_size=window_size)
    arr = np.concatenate([[3.7, 1e3, -2.2], np.full(20, 0.1)])
    for i, num in enumerate(arr):
        variance = running_stat.update(num).get()
        assert variance >= 0.0
        assert np.isclose(variance, np.var(arr[max(0, i - window_size + 1) : i + 1]), atol=1e-9)


def test_variance_meter_restarts_after_removing_every_value():
    from pysad.statistics import VarianceMeter

    stat = VarianceMeter()
    stat.update(3.0).update(5.0)
    stat.remove(3.0).remove(5.0)
    stat.update(1.0).update(2.0)

    assert stat.get() == 0.25
