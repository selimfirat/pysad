Evaluate a detector
===================

To evaluate a detector, you need a stream with labels: 1 for an anomaly and 0 for a normal point. The detector does not see the labels. A metric compares the scores with the labels while the stream runs.

Choose a metric
---------------

Some metrics need a threshold. A point with a score at or above the threshold raises an alert.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Metric
     - Use it to
   * - :class:`~pysad.evaluation.AUROCMetric`
     - Measure how well the scores rank anomalies above normal points. AUROC is the area under the ROC curve. A perfect ranking gets 1.0 and a random ranking gets 0.5. AUROC does not need a threshold.
   * - :class:`~pysad.evaluation.AUPRMetric`
     - Measure the ranking when anomalies are rare. AUPR is the area under the precision-recall curve. A random detector gets the fraction of anomalies in the stream, so compare AUPR only on the same data.
   * - :class:`~pysad.evaluation.PrecisionMetric`
     - Measure the fraction of alerts that are true anomalies, for a given threshold.
   * - :class:`~pysad.evaluation.RecallMetric`
     - Measure the fraction of anomalies that raise an alert, for a given threshold.

Measure several metrics at once
-------------------------------

Update every metric with the same score. Precision and recall need a threshold. In this example, the metrics get the calibrated score, a value between 0 and 1 (see :doc:`alerts`). As a result, the threshold 0.95 has the same meaning for every detector:

.. code-block:: python

   from pysad.evaluation import AUPRMetric, AUROCMetric, PrecisionMetric, RecallMetric
   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
   from pysad.utils import ArrayStreamer, Data

   X, y = Data().get_data("cardio.mat")

   model = LODA(random_state=0)
   calibrator = ConformalProbabilityCalibrator(window_size=1000)
   metrics = {
       "AUROC": AUROCMetric(),
       "AUPR": AUPRMetric(),
       "Precision": PrecisionMetric(threshold=0.95),
       "Recall": RecallMetric(threshold=0.95),
   }

   for x, label in ArrayStreamer().iter(X, y):
       calibrated = calibrator.fit_transform_partial(model.fit_score_partial(x))
       for metric in metrics.values():
           metric.update(label, calibrated)

   for name, metric in metrics.items():
       print(f"{name:>9}: {metric.get():.3f}")

.. code-block:: text

       AUROC: 0.902
        AUPR: 0.416
   Precision: 0.435
      Recall: 0.381

The AUROC of the calibrated scores is lower than the AUROC of the raw scores (0.951 in :doc:`/getting_started`). In ``cardio``, the 176 anomalies are the last points of the file. The calibrator compares each score with recent scores. During this final run of anomalies, most of the recent scores are from anomalies too.

A metric keeps every label and score that it gets, and ``get`` computes the metric over all of them. As a result, each ``get`` call takes longer as the stream grows. Call ``get`` at the end of the stream or at intervals, not for every point.

Follow the performance over time
--------------------------------

A metric over the full stream hides changes over time. For example, it hides a detector that gets worse after the data drifts (changes its distribution). To see the change, compute the metric on each window of the stream separately. A window is a block of consecutive points. Start a new metric every 300 points:

.. code-block:: python

   model = LODA(random_state=0)
   metric = AUROCMetric()
   per_window = []

   stream = ArrayStreamer(shuffle=True, random_state=0).iter(X, y)
   for t, (x, label) in enumerate(stream, start=1):
       metric.update(label, model.fit_score_partial(x))
       if t % 300 == 0:  # close the window and start a new metric
           per_window.append(round(metric.get(), 2))
           metric = AUROCMetric()

   print(per_window)

.. code-block:: text

   [0.89, 0.93, 0.92, 0.93, 0.92, 0.94]

The first window has the lowest AUROC, because the detector starts with no data and learns during this window. This example shuffles ``cardio`` because all its anomalies are at the end. Each window must contain at least one anomaly and one normal point. If a window does not, AUROC is not defined and ``get`` raises an error.

:class:`~pysad.evaluation.WindowedMetric` computes the same values and gives only their average, here 0.923. Use it to report one number that gives the same weight to each window of the stream. With AUPR, a window without anomalies gives 0 and lowers this average.

Report a stable result
----------------------

Randomized detectors give a different result for each seed (value of ``random_state``). Report the mean and the standard deviation over several seeds:

.. code-block:: python

   import numpy as np

   results = []
   for seed in range(5):
       model = LODA(random_state=seed)
       metric = AUROCMetric()
       for x, label in ArrayStreamer().iter(X, y):
           metric.update(label, model.fit_score_partial(x))
       results.append(metric.get())

   print(f"AUROC: {np.mean(results):.3f} ± {np.std(results):.3f}")

.. code-block:: text

   AUROC: 0.956 ± 0.008

To compare detectors, give every detector the same points in the same stream order. ``fit_score_partial`` scores each point before the detector learns from it, as :doc:`/concepts` explains.

Next steps
----------

- :doc:`choose_detector` shows how to compare several detectors.
- :doc:`production` shows how to prepare a detector for a live stream.
