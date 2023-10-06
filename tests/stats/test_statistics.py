from pysad.statistics.abs_statistic import AbsStatistic
from pysad.statistics.running_statistic import RunningStatistic


def test_all_zero_stats():
    import numpy as np
    from pysad.statistics import AbsStatistic
    from pysad.statistics import RunningStatistic
    from pysad.statistics import AverageMeter
    from pysad.statistics import CountMeter
    from pysad.statistics import MaxMeter
    from pysad.statistics import MedianMeter
    from pysad.statistics import MinMeter
    from pysad.statistics import SumMeter
    from pysad.statistics import SumSquaresMeter
    from pysad.statistics import VarianceMeter
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
        VarianceMeter: 0.0
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
            if i > 1: # for variance meter.
                assert np.isclose(stat.get(), val if val != "count" else i+1)
                assert np.isclose(abs_stat.get(), val if val != "count" else i+1)
                assert np.isclose(running_stat.get(), val if val != "count" else min(i+1, window_size))

                stat.remove(num)
                abs_stat.remove(num)
                assert np.isclose(stat.get(), prev_value)
                assert np.isclose(abs_stat.get(), abs(prev_value))
                stat.update(num)
                abs_stat.update(num)

            prev_value = stat.get()


def test_stats_with_batch_numpy():

    from pysad.statistics import AverageMeter
    from pysad.statistics import CountMeter
    from pysad.statistics import MaxMeter
    from pysad.statistics import MedianMeter
    from pysad.statistics import MinMeter
    from pysad.statistics import SumMeter
    from pysad.statistics import SumSquaresMeter
    from pysad.statistics import VarianceMeter
    import numpy as np
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
        VarianceMeter: np.var
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

            if i > 1: # for variance meter.
                assert np.isclose(stat.get(), val(arr[:i+1]))
                assert np.isclose(running_stat.get(), val(arr[max(0, i-window_size+1):i+1]))
                assert np.isclose(abs(stat.get()), abs_stat.get())

            stat.remove(num)
            abs_stat.remove(num)

            if i > 1:
                assert np.isclose(stat.get(), prev_value)
                assert np.isclose(abs_stat.get(), abs(prev_value))

            stat.update(num)
            abs_stat.update(num)

            prev_value = stat.get()
