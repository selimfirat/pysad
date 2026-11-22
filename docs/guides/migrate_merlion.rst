.. meta::
   :description: Migrate time series anomaly detection from archived Salesforce Merlion (March 2026) to PySAD on Python 3.11+ and NumPy 2 with SpectralResidual, SeasonalHybridESD, and ModelEnsemble.

Migrate from Salesforce Merlion
===============================

Salesforce archived ``salesforce/merlion`` in March 2026. Projects still depending on Merlion often get blocked on legacy Python/NumPy pins and unmaintained Cython extensions. PySAD requires only ``numpy``, ``scipy``, and ``scikit-learn``, runs natively on Python 3.11+ and NumPy 2, and ships built-in streaming detectors, ensembles, calibration, and prequential evaluation.

Scope and architectural differences
-----------------------------------

Merlion coupled forecasting models (``Prophet``, ``Sarima``, LightGBM) with anomaly detectors and Spark utilities. PySAD keeps dependencies minimal and focuses on **streaming anomaly detection**:

- **Online scoring by default**: PySAD scores each point before learning from it (``fit_score_partial``), avoiding train/test batch leakage while keeping memory bounded.
- **Keep external forecasters if needed**: For residual-based detection using online forecasters, wrap a River forecaster with :class:`~pysad.integrations.PredictiveAnomalyDetector`. For batch multi-horizon forecasting pipelines, see :doc:`../comparisons/darts`.
- **Single-observation NumPy vectors**: Pass 1-D arrays of shape ``(num_features,)`` (or 2-D arrays to batch ``fit_score``) instead of ``merlion.utils.TimeSeries`` / pandas MultiIndex frames (use :class:`~pysad.utils.PandasStreamer` to stream DataFrames directly).

API mapping
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Merlion component
     - PySAD equivalent
     - Notes
   * - ``SpectralResidual``
     - :class:`~pysad.models.SpectralResidual`
     - Sliding FFT spectral saliency scorer.
   * - ``StatThreshold`` / seasonal ETS
     - :class:`~pysad.models.SeasonalHybridESD` or :class:`~pysad.models.HoltWinters`
     - Handles daily/weekly cycles on univariate telemetry.
   * - ``WindStats`` / rolling Z-score
     - :class:`~pysad.models.MedianAbsoluteDeviation`
     - Sliding robust scale; pair with :class:`~pysad.transform.preprocessing.SeasonalTrendDecomposer` for seasonality.
   * - ``IsolationForest`` / tree detectors
     - :class:`~pysad.models.OnlineIsolationForest`, :class:`~pysad.models.HalfSpaceTrees`, or :class:`~pysad.models.IForestASD`
     - Incremental trees with bounded forgetting windows.
   * - Forecast-based detectors (``Prophet`` / ``ARIMA`` residuals)
     - :class:`~pysad.integrations.PredictiveAnomalyDetector`
     - Streams one-step forecast error from River time-series models.
   * - ``DetectorEnsemble``
     - :class:`~pysad.models.ModelEnsemble` or ``&`` operator
     - Combines detector pipelines with :mod:`pysad.transform.ensemble` combiners.
   * - ``ZScore`` / ``AggregateAlarms`` calibration
     - :class:`~pysad.transform.postprocessing.ZScorePostprocessor`, :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator`, :class:`~pysad.transform.postprocessing.ThresholdFilter`
     - Normalize, calibrate to ``[0, 1]``, and emit binary decisions.
   * - ``TSADMetric`` evaluation
     - :func:`~pysad.evaluation.evaluate_stream` and :mod:`pysad.evaluation`
     - Computes AUROC, AUPR, and windowed precision/recall/F1 prequentially.

Replace a Merlion detector ensemble and threshold
-------------------------------------------------

In Merlion, multivariate detectors are frequently stacked in a ``DetectorEnsemble`` followed by post-rule calibration. In PySAD, compose independent normalized branches with ``&`` (creating a :class:`~pysad.models.ModelEnsemble`) and append calibration and thresholding with ``|``:

.. code-block:: python

   import numpy as np

   from pysad.models import LODA, OnlineIsolationForest
   from pysad.transform.postprocessing import ThresholdFilter
   from pysad.transform.preprocessing import StandardScaler
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   rng = np.random.default_rng(3)
   X = rng.normal(size=(300, 3))
   X[275] = [5.5, -5.0, 4.5]

   ensemble = (
       (StandardScaler() | LODA(random_state=3))
       & (StandardScaler() | OnlineIsolationForest(num_trees=50, window_size=256, random_state=3))
   )
   pipeline = ensemble | ConformalProbabilityCalibrator(window_size=200) | ThresholdFilter(0.99)

   for step, x in enumerate(X):
       alert = pipeline.fit_predict_partial(x)
       if alert == 1 and step >= 150:
           print(f"ensemble alert at step {step}")

Prequential evaluation without ``TimeSeries`` wrappers
------------------------------------------------------

PySAD evaluates streaming models strictly prequentially—scoring each observation before fitting it—via :func:`~pysad.evaluation.evaluate_stream`:

.. code-block:: python

   import numpy as np

   from pysad.evaluation import AUPRMetric, AUROCMetric, evaluate_stream
   from pysad.models import SeasonalHybridESD

   rng = np.random.default_rng(1)
   t = np.arange(24 * 12)
   values = 20.0 + 4.0 * np.sin(2 * np.pi * t / 24) + rng.normal(scale=0.3, size=t.size)
   y = np.zeros(t.size, dtype=int)
   values[[220, 265]] += [6.0, -6.0]
   y[[220, 265]] = 1

   model = SeasonalHybridESD(period=24, window_size=24 * 7, direction="both", alpha=0.001)
   metrics = evaluate_stream(
       model,
       values[:, None],
       y,
       {"auroc": AUROCMetric(), "aupr": AUPRMetric()},
   )
   print({k: round(v, 3) for k, v in metrics.items() if v is not None})

Next steps
----------

- Compose preprocessors, ensembles, and calibrators in :doc:`pipelines`.
- Explore all windowed and classification metrics in :doc:`evaluate`.
- Stream directly from pandas DataFrames using :doc:`stream_data`.
