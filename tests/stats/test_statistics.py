from pysad.stats.abs_statistic import AbsStatistic


def test_all_zero_stats():
    import numpy as np
    from pysad.stats.average_meter import AverageMeter
    from pysad.stats.count_meter import CountMeter
    from pysad.stats.max_meter import MaxMeter
    from pysad.stats.median_meter import MedianMeter
    from pysad.stats.min_meter import MinMeter
    from pysad.stats.sum_meter import SumMeter
    from pysad.stats.sum_squares_meter import SumSquaresMeter
    from pysad.stats.variance_meter import VarianceMeter

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
        arr = np.zeros(num_items, dtype=np.float)
        prev_value = 0.0
        for i in range(arr.shape[0]):
            num = arr[i]
            stat.update(num)
            abs_stat.update(num)
            if i > 1: # for variance meter.
                assert np.isclose(stat.get(), val if val != "count" else i+1)
                assert np.isclose(abs_stat.get(), val if val != "count" else i+1)

                stat.remove(num)
                abs_stat.remove(num)
                assert np.isclose(stat.get(), prev_value)
                assert np.isclose(abs_stat.get(), abs(prev_value))
                stat.update(num)
                abs_stat.update(num)

            prev_value = stat.get()
