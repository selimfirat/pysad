.. meta::
   :description: Migrate from ADTK batch detection to PySAD streaming detection: map rule-based and statistical detectors, rolling windows, and label aggregation to PySAD models, calibration, and evaluation.

Migrate from ADTK to PySAD
==========================

`ADTK <https://github.com/arundo/adtk>`_ applies rule-based and statistical detectors to a whole pandas Series or DataFrame, or to a rolling window. It does not learn from one point at a time. PySAD scores each point when it arrives, then learns from it. No new dependency is needed: PySAD runs on numpy, scipy, and scikit-learn.

Scope and architectural differences
-----------------------------------

- **Whole series versus one point**: ADTK validates a series with ``validate_series`` and runs ``fit_detect`` on the whole series or on rolling windows. PySAD takes one vector of shape ``(num_features,)`` per ``fit_score_partial`` call.
- **Labels versus scores**: ADTK detectors return boolean anomalies directly. PySAD detectors return scores. Chain a postprocessor or a probability calibrator and a threshold to turn scores into alerts (see :doc:`alerts`).
- **Aggregation on labels**: ADTK combines boolean results with ``AndAggregation``, ``OrAggregation`` and ``MajorityVoteAggregation``. PySAD combines detector pipelines with the ``&`` (AND) and ``|`` (OR) operators, which build a :class:`~pysad.models.ModelEnsemble`, or combines scores with a score ensembler.
- **Pipeline**: ADTK chains detectors with ``DetectorPipeline``. PySAD chains transformers, detectors and postprocessors with ``|``.
- **Evaluation**: ADTK summarizes anomalies with ``summarize_anomalies``. PySAD measures prequential AUROC, AUPR, precision and recall with :func:`~pysad.evaluation.evaluate_stream`.

Use ADTK if you already have the whole series and want rule-based detectors without training. Use PySAD if points arrive one at a time, each point needs a score when it arrives, or you need calibration, ensembling, and prequential evaluation.

API mapping
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - ADTK
     - PySAD equivalent
     - Notes
   * - ``validate_series``
     - :class:`~pysad.utils.PandasStreamer` or :class:`~pysad.utils.ArrayStreamer`
     - Stream a DataFrame or array one row at a time.
   * - ``ThresholdAD(high, low)``
     - :class:`~pysad.transform.postprocessing.ThresholdFilter` on calibrated scores
     - ADTK flags values outside fixed bounds. PySAD calibrates scores first, then thresholds them.
   * - ``VolatilityAD(window_size, c)``
     - :class:`~pysad.models.MedianAbsoluteDeviation` with :class:`~pysad.transform.postprocessing.ThresholdFilter`
     - Sliding robust scale replaces rolling volatility.
   * - ``InterQuartileRangeAD(window_size)``
     - :class:`~pysad.transform.postprocessing.QuantileFilter`
     - Labels the top quantile of recent scores.
   * - ``GeneralizedESDTestAD(max_anomalies, alpha)``
     - :class:`~pysad.models.SeasonalESD`
     - Removes seasonality, then applies the generalized extreme studentized deviate test.
   * - ``PersistAD()``
     - No direct equivalent
     - ADTK flags a value equal to the previous value. Compare each point with the previous point in your loop.
   * - ``SeasonalAD(window_size, side)``
     - :class:`~pysad.models.SeasonalHybridESD` or :class:`~pysad.models.SeasonalESD`
     - Detects values that break the daily or weekly cycle.
   * - ``LevelShiftAD(window_size, c, side)``
     - :class:`~pysad.models.MedianAbsoluteDeviation` on first differences plus :class:`~pysad.transform.postprocessing.ThresholdFilter`
     - Or :class:`~pysad.models.DriftAdaptiveDetector` with :class:`~pysad.statistics.ADWINMeter` to reset on a shift.
   * - ``CustomRuleAD(filters)``
     - :class:`~pysad.transform.preprocessing.FunctionTransformer` plus :class:`~pysad.transform.postprocessing.ThresholdFilter`
     - Express custom rules as a transformer on each point.
   * - ``AndAggregation``, ``OrAggregation``, ``MajorityVoteAggregation``
     - :class:`~pysad.models.ModelEnsemble` with ``&`` and ``|``, or a score ensembler
     - ``&`` requires every branch to alert, ``|`` requires any branch to alert.
   * - ``DetectorPipeline``
     - ``|`` pipeline of transformers, detectors and postprocessors
     - PySAD pipelines run ``fit_score_partial`` on each stage in order.
   * - ``summarize_anomalies``
     - :func:`~pysad.evaluation.evaluate_stream` with :class:`~pysad.evaluation.AUROCMetric`, :class:`~pysad.evaluation.AUPRMetric`, :class:`~pysad.evaluation.PrecisionMetric` and :class:`~pysad.evaluation.RecallMetric`
     - Prequential metrics on labeled streams.

Replace ThresholdAD and SeasonalAD
----------------------------------

ADTK validates the series and runs two detectors on it:

.. code-block:: python

   import pandas as pd

   from adtk.data import validate_series
   from adtk.detector import ThresholdAD, SeasonalAD

   series = validate_series(pd.read_csv("metrics.csv", index_col=0, parse_dates=True))
   threshold_ad = ThresholdAD(high=15.0, low=5.0)
   seasonal_ad = SeasonalAD(window_size=24, side="both")
   anomalies = threshold_ad.fit_detect(series) + seasonal_ad.fit_detect(series)

The PySAD equivalent scores one point at a time, calibrates the scores, and thresholds them:

.. code-block:: python

   import numpy as np
   import pandas as pd

   from pysad.models import SeasonalHybridESD
   from pysad.transform.postprocessing import ThresholdFilter
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
   from pysad.utils import PandasStreamer

   df = pd.read_csv("metrics.csv", index_col=0, parse_dates=True)
   streamer = PandasStreamer()

   model = SeasonalHybridESD(period=24, window_size=24 * 7, direction="both", alpha=0.001)
   calibrator = ConformalProbabilityCalibrator(window_size=200)
   alerts = ThresholdFilter(threshold=0.99)

   for step, x in enumerate(streamer.iter(df.values)):
       score = model.fit_score_partial(x)
       calibrated = calibrator.fit_transform_partial(score)
       alert = alerts.transform_partial(calibrated)
       if alert == 1 and step >= 200:
           print(f"alert at step {step}")

``step >= 200`` suppresses alerts while the model and calibrator warm up.

Replace InterQuartileRangeAD on a rolling window
------------------------------------------------

ADTK flags values outside the rolling interquartile range:

.. code-block:: python

   from adtk.detector import InterQuartileRangeAD
   iqr_ad = InterQuartileRangeAD(window_size=50)
   anomalies = iqr_ad.fit_detect(series)

PySAD keeps a sliding robust scale with :class:`~pysad.models.MedianAbsoluteDeviation` and labels the most extreme scores:

.. code-block:: python

   import numpy as np

   from pysad.models import MedianAbsoluteDeviation
   from pysad.transform.postprocessing import ThresholdFilter

   model = MedianAbsoluteDeviation(window_size=50)
   alerts = ThresholdFilter(threshold=0.99)

   for x in stream:
       score = model.fit_score_partial(x)
       alert = alerts.transform_partial(score)

Replace label aggregation
-------------------------

ADTK combines detector results on boolean labels:

.. code-block:: python

   from adtk.aggregator import AndAggregation, OrAggregation
   anomalies = AndAggregation().aggregate([threshold_ad, seasonal_ad])

PySAD combines whole detector pipelines:

.. code-block:: python

   from pysad.models import LODA, OnlineIsolationForest
   from pysad.transform.preprocessing import StandardScaler

   ensemble = (
       (StandardScaler() | LODA(random_state=1))
       & (StandardScaler() | OnlineIsolationForest(num_trees=50, window_size=256, random_state=1))
   )

``&`` raises an alert only when every branch raises one. Use ``|`` to raise an alert when any branch raises one. Append calibration and a threshold with ``|``:

.. code-block:: python

   from pysad.transform.postprocessing import ThresholdFilter
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   pipeline = ensemble | ConformalProbabilityCalibrator(window_size=200) | ThresholdFilter(threshold=0.99)

Evaluate on labeled data
------------------------

ADTK summarizes detected anomalies after the run. PySAD evaluates prequentially, scoring each point before learning from it:

.. code-block:: python

   import numpy as np

   from pysad.evaluation import AUROCMetric, PrecisionMetric, evaluate_stream

   rng = np.random.default_rng(0)
   X = rng.normal(size=(400, 2))
   y = np.zeros(400, dtype=int)
   y[300:310] = 1

   metrics = evaluate_stream(
       model,
       X,
       y,
       {"auroc": AUROCMetric(), "precision": PrecisionMetric()},
   )

Next steps
----------

- Calibrate scores and pick an alert threshold in :doc:`alerts`.
- Run a PyOD batch detector on sliding windows in :doc:`pyod`.
- Stream CSV files and DataFrames in :doc:`stream_data`.
