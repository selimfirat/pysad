Evaluate a detector
===================

To evaluate a detector, you need a stream with labels: 1 for an anomaly and 0 for a normal point. The detector does not see the labels. A metric compares the scores with the labels while the stream runs.

Choose a metric
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Metric
     - Use it to
   * - :class:`~pysad.evaluation.AUROCMetric`
     - Measure how well the scores rank anomalies above normal points. 1.0 is perfect and 0.5 is random. It does not need a threshold.
   * - :class:`~pysad.evaluation.AUPRMetric`
     - Measure the ranking when anomalies are rare. A random detector gets the fraction of anomalies in the stream, so compare AUPR only on the same data.
   * - :class:`~pysad.evaluation.PrecisionMetric`
     - Measure the fraction of alerts that are true anomalies, for a given threshold.
   * - :class:`~pysad.evaluation.RecallMetric`
     - Measure the fraction of anomalies that raise an alert, for a given threshold.

Measure several metrics at once
-------------------------------

Update every metric with the same score. Precision and recall need a threshold. Here, they use the calibrated score, so the threshold 0.95 has the same meaning for every detector:

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

The AUROC of the calibrated scores is lower than that of the raw scores (0.951 in :doc:`/getting_started`). In ``cardio``, the 176 anomalies are the last points of the file. The calibrator compares each score with recent scores, and during this block most recent scores are anomalous too.

A metric keeps every label and score that it gets, and ``get`` computes the metric over all of them. Call ``get`` at the end of the stream or at intervals, not for every point.

Follow the performance over time
--------------------------------

A metric over the full stream hides changes over time, for example a detector that gets worse after the data drifts. :class:`~pysad.evaluation.WindowedMetric` computes a metric on each window of the stream on its own and gives the average over the windows:

.. code-block:: python

   from pysad.evaluation import AUROCMetric, WindowedMetric

   model = LODA(random_state=0)
   metric = WindowedMetric(AUROCMetric, window_size=300)

   for x, label in ArrayStreamer(shuffle=True, random_state=0).iter(X, y):
       metric.update(label, model.fit_score_partial(x))

   print(f"windowed AUROC: {metric.get():.3f}")

Each window must contain at least one anomaly and one normal point. Otherwise, AUROC is not defined and the metric raises an error. AUPR gives 0 for a window without anomalies and lowers the average. This example shuffles ``cardio`` because all its anomalies are at the end.

Report a stable result
----------------------

Randomized detectors give a different result for each seed. Report the mean and the standard deviation over several seeds:

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

To compare detectors, use the same stream in the same order, and the same evaluation order for every detector. :doc:`/concepts` explains why the order matters.
