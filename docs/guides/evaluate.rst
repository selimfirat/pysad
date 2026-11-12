Evaluate a detector
===================

Evaluate scores against labels: 1 for an anomaly, 0 for normal. Only the metrics use labels; the detector learns without them.

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
   * - :class:`~pysad.evaluation.F1Metric`
     - Measure the harmonic mean of precision and recall for a given threshold. Returns 0.0 for an empty stream or when there are no true or predicted anomalies.
   * - :class:`~pysad.evaluation.MatthewsCorrelationMetric`
     - Measure binary predictions when class sizes differ. It uses all four confusion counts and supports a score threshold.
   * - :class:`~pysad.evaluation.RecallMetric`
     - Measure the fraction of anomalies that raise an alert, for a given threshold.
   * - :class:`~pysad.evaluation.BalancedAccuracyMetric`
     - Give normal points and anomalies equal weight through their mean recall. Set a threshold for scores, or pass 0/1 predictions.

Measure several metrics at once
-------------------------------

:func:`~pysad.evaluation.evaluate_stream` evaluates raw detector scores. It scores before learning, updates each metric, and returns their final values.

``X`` and ``y`` accept arrays, lists or single-use generators and are consumed once in order. Instances have shape ``(num_features,)``; labels are 1 for anomalies or 0 for normal points. Unequal lengths raise ``ValueError`` after retaining earlier model and metric updates.

These generators avoid copying the arrays. The evaluation loop does not retain the stream, though metrics may retain labels and scores:

.. code-block:: python

   from pysad.evaluation import AUPRMetric, AUROCMetric, F1Metric, evaluate_stream
   from pysad.models import LODA
   from pysad.utils import Data

   X, y = Data().get_data("cardio.mat")

   results = evaluate_stream(
       model=LODA(random_state=0),
       X=(x for x in X),
       y=(int(label) for label in y),
       metrics={
           "AUROC": AUROCMetric(),
           "AUPR": AUPRMetric(),
           "F1": F1Metric(threshold=3.0),
       },
   )

The F1 threshold 3.0 is illustrative; choose yours on prior validation data. Equality counts as anomalous. With ``threshold=None``, pass binary predictions.

For transformed scores, update all metrics in your own loop. Here conformal scores range from 0 to 1, and precision and recall use a 0.95 rank threshold; see :doc:`alerts`:

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

``PrecisionMetric``, ``RecallMetric``, ``F1Metric``, ``BalancedAccuracyMetric`` and :class:`~pysad.evaluation.MatthewsCorrelationMetric` retain four confusion counts, giving constant update and read costs. ``AUROCMetric`` and ``AUPRMetric`` retain all labels and scores; their ``get`` cost grows with the stream. Read them periodically or at the end.

Follow the performance over time
--------------------------------

Full-stream metrics can hide performance changes after drift. Evaluate consecutive windows separately by starting a new metric every 300 points:

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

The first window includes the detector’s warm-up and has the lowest AUROC. We shuffle ``cardio`` because its anomalies are all at the end. AUROC requires both classes in each window; otherwise ``get`` raises an error.

:class:`~pysad.evaluation.WindowedMetric` returns their equally weighted average, here 0.923. For AUPR, a window without anomalies contributes 0.

Track recent performance
------------------------

:class:`~pysad.evaluation.RollingMetric` evaluates only the most recent label-score pairs.
Each update drops the oldest pair when the window is full.
It differs from ``WindowedMetric``, which averages results from consecutive windows that do not overlap.

.. code-block:: python

   from pysad.evaluation import RecallMetric, RollingMetric

   metric = RollingMetric(RecallMetric, window_size=100, threshold=0.5)
   print(metric.get())  # None, because the window is empty.
   metric.update(1, 0.8)
   metric.update(1, 0.2)
   print(metric.get())  # 0.5, from the two available pairs.

An empty window returns ``None`` for every metric class.
A nonempty window follows the selected metric's behavior, including errors for undefined results.
For example, ``AUROCMetric`` needs both label classes in the recent window.
Updates take O(1) time, and storage holds at most ``window_size`` pairs.
Each ``get()`` creates a fresh metric, replays the window, and reads its result.
AUROC and AUPR reads take O(window_size * log(window_size)) time because they sort scores.
Read less often to reduce this cost.
See ``examples/example_rolling_metric.py`` for a complete stream example.

Report a stable result
----------------------

For randomized detectors, report the mean and standard deviation over several ``random_state`` seeds:

.. code-block:: python

   import numpy as np

   results = []
   for seed in range(5):
       model = LODA(random_state=seed)
       auroc = evaluate_stream(model=model, X=X, y=y, metrics={"AUROC": AUROCMetric()})["AUROC"]
       results.append(auroc)

   print(f"AUROC: {np.mean(results):.3f} ± {np.std(results):.3f}")

.. code-block:: text

   AUROC: 0.956 ± 0.008

Compare detectors on identical points in identical order. ``evaluate_stream`` scores before learning; see :doc:`/concepts`.

Next steps
----------

- :doc:`choose_detector` shows how to compare several detectors.
- :doc:`production` shows how to prepare a detector for a live stream.
