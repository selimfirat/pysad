.. meta::
   :description: Migrate from PyOD batch anomaly detection to PySAD prequential streaming: map PyOD fit, decision_function and predict to fit_score_partial, calibration, alerts, and prequential evaluation.

Migrate from PyOD to PySAD
==========================

`PyOD <https://github.com/yzhao062/pyod>`_ trains a detector on a whole dataset, then scores new points with ``decision_function`` or labels them with ``predict``. It does not learn from one point at a time. PySAD scores each point when it arrives, then learns from it. PySAD runs on numpy, scipy, and scikit-learn. PyOD is optional and only needed to keep running PyOD detectors during migration.

Scope and architectural differences
-----------------------------------

- **Whole dataset versus one point**: PyOD calls ``fit`` on the whole training set, then ``decision_function`` on new points. PySAD takes one vector of shape ``(num_features,)`` per ``fit_score_partial`` call: score first, then learn.
- **Labels versus scores**: PyOD ``predict`` returns anomaly labels directly. PySAD detectors return scores. Chain a probability calibrator and a threshold to turn scores into alerts (see :doc:`alerts`).
- **Memory**: PyOD keeps the training data or a model that grows with it. PySAD models keep a window or summary statistics, so memory stays bounded as the stream grows.
- **Retraining**: PyOD retrains on new data by calling ``fit`` again on a larger set. PySAD updates in place on each point. Use :class:`~pysad.integrations.ReferenceWindowModel` to retrain a PyOD detector on a sliding window during migration.
- **Evaluation**: PyOD scores a held-out test set with ROC AUC or precision at rank n. PySAD measures prequential AUROC, AUPR, precision and recall with :func:`~pysad.evaluation.evaluate_stream`, scoring each point before learning from it.

Use PyOD if all your data is available before you train and you score a static test set. Use PySAD if points arrive one at a time, each point needs a score when it arrives, or you need calibration, ensembling, and prequential evaluation.

API mapping
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - PyOD
     - PySAD equivalent
     - Notes
   * - ``fit(X)``
     - :meth:`~pysad.core.BaseModel.fit_partial`
     - Call once per point, after scoring it.
   * - ``decision_function(X)``
     - :meth:`~pysad.core.BaseModel.score_partial`
     - Returns one Python float per point. Higher means more anomalous.
   * - ``predict(X)``
     - probability calibrator plus :class:`~pysad.transform.postprocessing.ThresholdFilter`
     - PySAD calibrates scores, then thresholds them into 0.0 or 1.0 alerts.
   * - Batch detector such as IForest, KNN or LOF
     - :class:`~pysad.models.OnlineIsolationForest`, :class:`~pysad.models.IncrementalLOF`, :class:`~pysad.models.LODA`, or :class:`~pysad.integrations.ReferenceWindowModel`
     - Native streaming detectors keep bounded memory. The wrapper keeps running a PyOD detector on recent windows.
   * - ``predict_proba``
     - :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator` or :class:`~pysad.transform.probability_calibration.SPOTProbabilityCalibrator`
     - Online calibration updated with each score.
   * - ROC AUC or precision at n on a test set
     - :func:`~pysad.evaluation.evaluate_stream` with :class:`~pysad.evaluation.AUROCMetric`, :class:`~pysad.evaluation.AUPRMetric`, :class:`~pysad.evaluation.PrecisionMetric` and :class:`~pysad.evaluation.RecallMetric`
     - Prequential metrics on labeled streams.

Keep running PyOD on sliding windows
------------------------------------

Use :class:`~pysad.integrations.ReferenceWindowModel` to retrain a PyOD batch detector every ``sliding_size`` points, or :class:`~pysad.integrations.OneFitModel` to fit once on initial data and keep the baseline fixed. See :doc:`pyod` for full working examples of both wrappers.

Replace batch scoring with native streaming detectors
-----------------------------------------------------

PyOD trains on the whole set, then scores. PySAD scores each point, then learns from it:

.. code-block:: python

   import numpy as np

   from pysad.models import OnlineIsolationForest
   from pysad.transform.preprocessing import StandardScaler
   from pysad.transform.postprocessing import ThresholdFilter
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   rng = np.random.default_rng(5)
   stream = rng.normal(loc=0.0, scale=1.0, size=(300, 6))
   stream[270] += 6.0

   pipeline = (
       StandardScaler()
       | OnlineIsolationForest(num_trees=100, window_size=256, random_state=5)
       | ConformalProbabilityCalibrator(window_size=200)
       | ThresholdFilter(threshold=0.99)
   )

   for step, x in enumerate(stream):
       alert = pipeline.fit_predict_partial(x)
       if alert == 1 and step >= 200:
           print(f"alert at step {step}")

``step >= 200`` suppresses alerts while the model and calibrator warm up.

Evaluate on labeled data
------------------------

PyOD scores a held-out test set after training. PySAD evaluates prequentially, scoring each point before learning from it:

.. code-block:: python

   import numpy as np

   from pysad.evaluation import AUROCMetric, evaluate_stream
   from pysad.models import LODA

   rng = np.random.default_rng(0)
   X = rng.normal(size=(400, 2))
   X[300:320] += 5.0
   y = np.zeros(400, dtype=int)
   y[300:320] = 1

   model = LODA(random_state=0)

   metrics = evaluate_stream(
       model,
       X,
       y,
       {"auroc": AUROCMetric()},
   )

Next steps
----------

- :doc:`pyod` runs PyOD batch detectors on sliding windows.
- :doc:`alerts` calibrates scores and picks an alert threshold.
- :doc:`evaluate` chooses a metric and follows performance over time.
- :doc:`stream_data` streams CSV files and DataFrames one row at a time.
