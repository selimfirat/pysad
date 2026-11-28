.. meta::
   :description: Migrate from River streaming anomaly detection to PySAD: map learn_one, score_one, progressive_val_score, pipelines, filters and metrics to PySAD fit_partial, score_partial, evaluate_stream, pipelines, postprocessors and prequential metrics.

Migrate from River to PySAD
============================

`River <https://riverml.xyz/>`_ does machine learning on streams. Like PySAD, it learns from one point at a time. River scores each point with ``score_one`` and learns with ``learn_one``. PySAD scores each point with ``score_partial`` and learns with ``fit_partial``. PySAD runs on numpy, scipy, and scikit-learn. No new dependency is needed: River is optional and only needed to keep running River detectors during migration.

Scope and architectural differences
-----------------------------------

- **Feature dictionaries versus arrays**: River passes one instance as a dict of feature names to values. PySAD passes one instance as a NumPy array of shape ``(num_features,)``.
- **Method names**: River calls ``score_one(x)`` then ``learn_one(x)``. PySAD calls ``score_partial(x)`` then ``fit_partial(x)``, or ``fit_score_partial(x)`` to do both in one call.
- **Prequential order**: River's ``score_one`` does not learn. You must call ``learn_one`` yourself after scoring. PySAD's ``fit_score_partial`` scores first, then learns, in a single call.
- **Labels versus scores**: River anomaly detectors return scores. PySAD detectors return scores. Chain a probability calibrator and a threshold to turn scores into alerts (see :doc:`alerts`).
- **Pipelines**: River chains stages with ``|``. PySAD chains transformers, detectors and postprocessors with ``|`` through :class:`~pysad.models.StreamPipeline`.
- **Evaluation**: River uses ``progressive_val_score`` with an approximate ROC AUC. PySAD measures prequential AUROC, AUPR, precision and recall with :func:`~pysad.evaluation.evaluate_stream`, scoring each point before learning from it.

Use River if anomaly detection is one step of a larger River pipeline. Use PySAD if points arrive as arrays, each point needs a score when it arrives, or you need calibration, ensembling, and prequential evaluation.

API mapping
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - River
     - PySAD equivalent
     - Notes
   * - ``score_one(x)``
     - :meth:`~pysad.core.BaseModel.score_partial`
     - Returns one Python float per point. Higher means more anomalous.
   * - ``learn_one(x)``
     - :meth:`~pysad.core.BaseModel.fit_partial`
     - Call once per point, after scoring it.
   * - ``score_one`` then ``learn_one``
     - :meth:`~pysad.core.BaseModel.fit_score_partial`
     - Scores first, then learns, in one call.
   * - ``HalfSpaceTrees``
     - :class:`~pysad.models.HalfSpaceTrees`
     - Same streaming detector, same bounded memory.
   * - ``LODA``
     - :class:`~pysad.models.LODA`
     - Same streaming detector, same bounded memory.
   * - ``LocalOutlierFactor``
     - :class:`~pysad.models.IncrementalLOF`
     - PySAD keeps a sliding window of recent points.
   * - ``OneClassSVM``
     - :class:`~pysad.models.OnlineOneClassSVM`
     - Online version of one-class SVM.
   * - ``StandardAbsoluteDeviation``
     - :class:`~pysad.models.StandardAbsoluteDeviation`
     - Same streaming detector, same bounded memory.
   * - ``PredictiveAnomalyDetection``
     - :class:`~pysad.models.StreamingAutoregressiveDetector`
     - Predicts the next point and scores the error.
   * - ``progressive_val_score``
     - :func:`~pysad.evaluation.evaluate_stream`
     - Prequential metrics on labeled streams.
   * - ``ThresholdFilter``, ``QuantileFilter``
     - :class:`~pysad.transform.postprocessing.ThresholdFilter`, :class:`~pysad.transform.postprocessing.QuantileFilter`
     - PySAD applies them to calibrated scores after the detector.
   * - Pipeline with ``|``
     - :class:`~pysad.models.StreamPipeline` with ``|``
     - PySAD pipelines run ``fit_score_partial`` on each stage in order.

Replace a River anomaly detection loop
--------------------------------------

River scores a dict and learns from it:

.. code-block:: python

   from river import anomaly

   model = anomaly.HalfSpaceTrees(seed=0)
   x = {"feature_0": 1.0, "feature_1": 2.0}
   score = model.score_one(x)
   model.learn_one(x)

PySAD scores an array, then learns from it:

.. code-block:: python

   import numpy as np

   from pysad.models import HalfSpaceTrees

   model = HalfSpaceTrees(
       feature_mins=np.array([0.0, 0.0]),
       feature_maxes=np.array([10.0, 10.0]),
       random_state=0,
   )
   x = np.array([1.0, 2.0])
   score = model.fit_score_partial(x)

``fit_score_partial`` returns the score and updates the model in one call.

Replace a River pipeline
------------------------

River chains a scaler and a detector:

.. code-block:: python

   from river import anomaly, preprocessing

   pipeline = preprocessing.StandardScaler() | anomaly.HalfSpaceTrees(seed=0)
   x = {"feature_0": 1.0, "feature_1": 2.0}
   score = pipeline.score_one(x)
   pipeline.learn_one(x)

PySAD chains a scaler and a detector through a stream pipeline:

.. code-block:: python

   import numpy as np

   from pysad.models import HalfSpaceTrees, StreamPipeline
   from pysad.transform.preprocessing import StandardScaler

   pipeline = StandardScaler() | HalfSpaceTrees(
       feature_mins=np.array([0.0, 0.0]),
       feature_maxes=np.array([10.0, 10.0]),
       random_state=0,
   )
   x = np.array([1.0, 2.0])
   score = pipeline.fit_score_partial(x)

The pipeline scales the point, scores it, then learns from it, in that order.

Replace River filters with PySAD postprocessors
-----------------------------------------------

River flags extreme scores with a filter:

.. code-block:: python

   from river import anomaly

   model = anomaly.QuantileFilter(anomaly.HalfSpaceTrees(seed=0), q=0.99)
   x = {"feature_0": 1.0, "feature_1": 2.0}
   score = model.score_one(x)
   model.learn_one(x)
   alert = model.classify(score)

PySAD calibrates the score, then thresholds it:

.. code-block:: python

   import numpy as np

   from pysad.models import HalfSpaceTrees
   from pysad.transform.postprocessing import ThresholdFilter
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   model = HalfSpaceTrees(
       feature_mins=np.array([0.0, 0.0]),
       feature_maxes=np.array([10.0, 10.0]),
       random_state=0,
   )
   calibrator = ConformalProbabilityCalibrator(window_size=200)
   alerts = ThresholdFilter(threshold=0.99)

   x = np.array([1.0, 2.0])
   score = model.fit_score_partial(x)
   calibrated = calibrator.fit_transform_partial(score)
   alert = alerts.transform_partial(calibrated)

Evaluate on labeled data
------------------------

River evaluates with ``progressive_val_score``:

.. code-block:: python

   from river import evaluate, anomaly, metrics

   metric = metrics.ROCAUC()
   evaluate.progressive_val_score(dataset, anomaly.HalfSpaceTrees(seed=0), metric)

PySAD evaluates prequentially, scoring each point before learning from it:

.. code-block:: python

   import numpy as np

   from pysad.evaluation import AUROCMetric, evaluate_stream
   from pysad.models import HalfSpaceTrees

   rng = np.random.default_rng(0)
   X = rng.normal(size=(400, 2))
   X[300:320] += 5.0
   y = np.zeros(400, dtype=int)
   y[300:320] = 1

   model = HalfSpaceTrees(
       feature_mins=np.full(2, -5.0),
       feature_maxes=np.full(2, 5.0),
       random_state=0,
   )

   metrics = evaluate_stream(
       model,
       X,
       y,
       {"auroc": AUROCMetric()},
   )

Next steps
----------

- Run PySAD detectors inside River pipelines in :doc:`river_interoperability`.
- Calibrate scores and pick an alert threshold in :doc:`alerts`.
- Stream CSV files and DataFrames in :doc:`stream_data`.
- Run PyOD batch detectors on sliding windows in :doc:`pyod`.
