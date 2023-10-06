
def helper_test_all_metrics(metric_classes, y_true, y_pred, ignore_nonempty_last):
    import numpy as np
    from pysad.evaluation import WindowedMetric

    for metric_cls, val in metric_classes.items():
        metric = WindowedMetric(metric_cls, 25, ignore_nonempty_last)

        for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
            metric.update(yt, yp)
            if i > 0:
                assert np.isclose(metric.get(), val)


def test_all_correct():
    from pysad.evaluation import PrecisionMetric, AUPRMetric, AUROCMetric, RecallMetric
    import numpy as np
    metric_classes = [
        PrecisionMetric,
        RecallMetric,
        AUPRMetric,
        AUROCMetric
    ]
    metric_classes = { metric_cls: 1.0 for metric_cls in metric_classes }
    y_true = np.random.randint(0, 2, size=(25,), dtype=np.int32)
    y_true[0] = 1
    y_true[1] = 0
    y_pred = y_true.copy()

    for ignore_nonempty_last in [True, False]:
        helper_test_all_metrics(metric_classes, y_true, y_pred, ignore_nonempty_last)


def test_none_correct():
    from pysad.evaluation import PrecisionMetric, AUPRMetric, AUROCMetric, RecallMetric
    import numpy as np

    metric_classes = {
        PrecisionMetric: 0.0,
        #AUPRMetric: 0.5
        AUROCMetric: 0.0,
        RecallMetric: 0.0
    }
    y_true = np.random.randint(0, 2, size=(25,), dtype=np.int32)
    y_true[0] = 1
    y_true[1] = 0
    y_pred = 1 - y_true.copy()

    for ignore_nonempty_last in [True, False]:
        helper_test_all_metrics(metric_classes, y_true, y_pred, ignore_nonempty_last)
