Evaluation
==========

.. module:: pysad.evaluation

This module holds streaming metrics for anomaly detection on data streams. Each metric records label-score pairs one point at a time with ``update`` and reports its current value with ``get``. Labels use 1 for anomalies and 0 for normal instances. Detectors learn without the labels.

A detector scores each instance before it learns from it. This is prequential evaluation, and it is what :func:`~pysad.evaluation.evaluate_stream` implements.

Prequential evaluation
----------------------

:func:`~pysad.evaluation.evaluate_stream` evaluates a model on a labeled stream. For each instance, the model scores the instance with its current state, and then it learns from the instance. The model never receives the labels. Every metric records each label-score pair.

An optional calibrator or postprocessor can transform each score before the metrics record it. The postprocessor learns from the score in the same pass, so calibration stays prequential. To evaluate a detector together with a calibrator, pass the calibrator as ``postprocessor`` instead of wrapping the detector in a pipeline:

.. code-block:: python

   from pysad.evaluation import AUROCMetric, evaluate_stream
   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
   from pysad.utils import Data

   X, y = Data().get_data("cardio.mat")

   results = evaluate_stream(
       model=LODA(random_state=0),
       X=X,
       y=y,
       metrics={"AUROC": AUROCMetric()},
       postprocessor=ConformalProbabilityCalibrator(window_size=1000),
   )

The function accepts arrays, lists, or single-use generators and consumes them once in stream order. If ``X`` and ``y`` have unequal lengths, it raises ``ValueError``. Model and metric updates from earlier pairs are retained.

Configuration schema
--------------------

:class:`~pysad.evaluation.config.EvaluationConfig` is the shared configuration schema for a prequential evaluation run. It bundles a streaming detector, an optional calibrator or postprocessor, and the metrics to record:

.. code-block:: python

   from pysad.evaluation.config import EvaluationConfig
   from pysad.evaluation import AUROCMetric, evaluate_stream
   from pysad.models import LODA

   config = EvaluationConfig(
       model=LODA(random_state=0),
       metrics={"AUROC": AUROCMetric()},
   )
   results = evaluate_stream(
       model=config.model,
       X=X,
       y=y,
       metrics=config.metrics,
       postprocessor=config.postprocessor,
   )

See ``examples/example_evaluate_stream.py`` for an example with several metrics at once.

Classification metrics
----------------------

These metrics compare labels with predictions or scores:

- :class:`~pysad.evaluation.AUROCMetric` measures how well scores rank anomalies above normal points. It is the area under the ROC curve. A perfect ranking gets 1.0 and a random ranking gets 0.5. It does not need a threshold. ``get`` raises ``ValueError`` when only one class has been observed.
- :class:`~pysad.evaluation.AUPRMetric` measures the area under the precision-recall curve. Use it when anomalies are rare. A random detector gets the fraction of anomalies in the stream.
- :class:`~pysad.evaluation.PrecisionMetric` measures the fraction of alerts that are true anomalies. With ``threshold=None``, predictions must be 0 or 1. With a threshold, scores at or above it predict an anomaly.
- :class:`~pysad.evaluation.RecallMetric` measures the fraction of anomalies that raise an alert. It supports the same ``threshold`` option as precision.
- :class:`~pysad.evaluation.F1Metric` measures the harmonic mean of precision and recall. It returns 0.0 for an empty stream or a zero denominator.
- :class:`~pysad.evaluation.BalancedAccuracyMetric` gives normal points and anomalies equal weight through the mean of sensitivity and specificity.
- :class:`~pysad.evaluation.MatthewsCorrelationMetric` measures binary predictions with all four confusion counts. Its result ranges from -1 to 1.

The confusion-matrix metrics keep four counts, so updates and reads take constant time. AUROC and AUPR retain all labels and scores, so their read cost grows with the stream.

Probability metrics
-------------------

- :class:`~pysad.evaluation.BrierScoreMetric` measures the mean squared error of predicted anomaly probabilities. Lower scores mean better probabilities. An empty stream returns 0.0.
- :class:`~pysad.evaluation.ExpectedCalibrationErrorMetric` measures calibration. It compares each probability bin's mean probability with its fraction of anomalies. Lower scores mean better calibration.
- :class:`~pysad.evaluation.LogLossMetric` is the prequential logarithmic loss :cite:`cesabianchi2001worst`. It clips probabilities to ``[eps, 1 - eps]`` for numerical stability. Updates take constant time and memory. An empty stream returns None.

Rolling metrics
---------------

:class:`~pysad.evaluation.RollingMetric` evaluates a metric on the most recent label-score pairs. Each update adds one pair and drops the oldest pair when the window is full. An empty window returns ``None``. A partial window uses all available pairs.

.. code-block:: python

   from pysad.evaluation import RecallMetric, RollingMetric

   metric = RollingMetric(metric_cls=RecallMetric, window_size=100, threshold=0.5)
   print(metric.get())  # None, because the window is empty.
   metric.update(1, 0.8)
   metric.update(1, 0.2)
   print(metric.get())  # 0.5, from the two available pairs.

Storage uses O(window_size) pairs, and updates take O(1) time. Each ``get()`` creates a fresh metric, replays the window, and reads its result. AUROC and AUPR reads take O(window_size * log(window_size)) time because they sort scores. Read less often to reduce this cost.

Unlike :class:`~pysad.evaluation.WindowedMetric`, which averages results from consecutive windows that do not overlap, a rolling window always covers the most recent pairs.

See ``examples/example_rolling_metric.py`` for a complete stream example.

Windowed metrics
----------------

:class:`~pysad.evaluation.WindowedMetric` evaluates a metric on consecutive windows of the stream and averages the window results :cite:`xstream,gokcesu2017online`. It creates a new metric instance after every ``window_size`` instances. Windows that contain only one class are skipped and do not contribute to the average.

.. code-block:: python

   from pysad.evaluation import AUROCMetric, WindowedMetric
   from pysad.models import xStream
   from pysad.utils import ArrayStreamer, Data

   X, y = Data().get_data("arrhythmia.mat")

   model = xStream(random_state=0)
   metric = AUROCMetric()
   windowed_metric = WindowedMetric(metric_cls=AUROCMetric, window_size=100)

   for x, label in ArrayStreamer().iter(X, y):
       score = model.fit_score_partial(x)
       metric.update(label, score)
       windowed_metric.update(label, score)

   print(f"AUROC: {metric.get():.3f}")
   print(f"Windowed AUROC: {windowed_metric.get():.3f}")

Set ``ignore_nonempty_last=False`` to include the incomplete last window in the average when it contains both classes. The empty last window is always ignored.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated
   :kind: classes
   :tail: BaseSKLearnMetric

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :kind: functions
