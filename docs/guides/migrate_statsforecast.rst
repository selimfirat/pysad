.. meta::
   :description: Migrate time series forecasting and anomaly detection from StatsForecast / Nixtla to PySAD streaming pipelines on Python 3.11+, with no new dependencies.

Migrate from StatsForecast and Nixtla to PySAD
===============================================

`StatsForecast <https://github.com/Nixtla/statsforecast>`_ is a forecasting library from Nixtla. It fits statistical models such as AutoARIMA, AutoETS, MSTL, Theta, and Croston on a whole series, and it forecasts future steps. Nixtla also publishes neuralforecast and mlforecast, which are batch forecasting libraries too. StatsForecast detects anomalies by comparing the forecast with the observed value and flagging large errors. It does not learn from one point at a time.

PySAD scores each point before it learns from it. It keeps bounded memory, and it needs only ``numpy``, ``scipy``, and ``scikit-learn`` on Python 3.11+. No new dependency is required to move a forecasting or anomaly workflow to PySAD.

Scope and architectural differences
-----------------------------------

- **Batch forecasting versus streaming scores**: StatsForecast learns from a complete series and then forecasts. PySAD scores each incoming point with ``fit_score_partial`` and then updates its state, so every point gets a score when it arrives.
- **No native anomaly detectors**: StatsForecast has forecasting models only. Anomalies come from forecast errors that the user thresholds. PySAD ships univariate and multivariate streaming detectors, probability calibrators, score ensemblers, and prequential evaluation.
- **Keep StatsForecast for batch jobs**: If you already run AutoARIMA or AutoETS on a complete series overnight, keep that job. Move to PySAD when values arrive one at a time or when you need calibrated streaming alerts.
- **Input format**: StatsForecast takes a long ``pandas.DataFrame`` with columns such as ``unique_id``, ``ds``, and ``y``. PySAD takes one NumPy vector of shape ``(num_features,)`` per point. Use :class:`~pysad.utils.PandasStreamer` to stream a DataFrame row by row.

API mapping
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - StatsForecast / Nixtla component
     - PySAD equivalent
     - Notes
   * - AutoARIMA, AutoETS, MSTL, Theta, Croston (batch forecast)
     - Keep StatsForecast offline for batch forecasts of a whole series
     - No PySAD equivalent is needed for a one-off batch forecast.
   * - One-step forecast error on a stream
     - :class:`~pysad.integrations.PredictiveAnomalyDetector` with a River forecaster
     - Wraps a River forecaster such as SNARIMAX and scores the absolute forecast error for each new point. Install River with ``pip install river``.
   * - Thresholded forecast errors
     - :class:`~pysad.models.SeasonalHybridESD`, :class:`~pysad.models.HoltWinters`, :class:`~pysad.models.SpectralResidual`, :class:`~pysad.models.MedianAbsoluteDeviation`
     - Streaming detectors that score each value as it arrives on univariate telemetry.
   * - Calibrated alert threshold
     - :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator`, :class:`~pysad.transform.probability_calibration.SPOTProbabilityCalibrator`, :class:`~pysad.transform.postprocessing.ThresholdFilter`
     - Map scores to tail probabilities or ranks in ``[0, 1]``, then apply a fixed cutoff.
   * - Backtesting and cross-validation
     - :func:`~pysad.evaluation.evaluate_stream` with :class:`~pysad.evaluation.AUROCMetric`
     - Computes AUROC and AUPR prequentially: each point is scored before the model learns it.
   * - Forecast or model combinations
     - :mod:`pysad.transform.ensemble` or :class:`~pysad.models.ModelEnsemble`
     - Combines the scores of several detectors or pipelines.
   * - nixtla_client or the Nixtla cloud forecast API
     - Self-hosted PySAD pipelines
     - Runs in your own process with no external service and no new dependency.

Replace forecast-error detection on a stream
---------------------------------------------

StatsForecast users often fit AutoETS or AutoARIMA on history, forecast the next value, and flag points whose error exceeds a fixed bound. On a stream, replace that loop with a streaming detector and a calibrator. This example uses only PySAD and NumPy:

.. code-block:: python

   import numpy as np

   from pysad.models import SeasonalHybridESD
   from pysad.transform.postprocessing import ThresholdFilter
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   rng = np.random.default_rng(0)
   t = np.arange(24 * 14)
   signal = 10.0 + 2.0 * np.sin(2 * np.pi * t / 24) + rng.normal(scale=0.2, size=t.size)
   signal[300:306] = 16.0  # anomaly burst

   pipeline = (
       SeasonalHybridESD(period=24, window_size=24 * 7, direction="both", alpha=0.001)
       | ConformalProbabilityCalibrator(window_size=200)
       | ThresholdFilter(threshold=0.99)
   )

   for step, value in enumerate(signal):
       alert = pipeline.fit_predict_partial(np.array([value]))
       if alert == 1 and step >= 250:
           print(f"anomaly at step {step}, value={value:.2f}")

It prints:

.. code-block:: text

   anomaly at step 300, value=16.00
   anomaly at step 301, value=16.00
   anomaly at step 302, value=16.00
   anomaly at step 303, value=16.00
   anomaly at step 304, value=16.00
   anomaly at step 305, value=16.00

The detector scores each point before it learns from it, so the burst is flagged as it arrives. ``fit_score_partial`` gives the same scores without updating the model.

Score one-step forecast errors online
-------------------------------------

If you want to keep a statistical forecaster on the stream, wrap a River forecaster with :class:`~pysad.integrations.PredictiveAnomalyDetector`. The detector scores each value by its deviation from the forecast, then learns the value. Install River with ``pip install river``:

.. code-block:: python

   import numpy as np

   from pysad.integrations import PredictiveAnomalyDetector
   from river.time_series import SNARIMAX

   rng = np.random.default_rng(1)
   stream = 50.0 + rng.normal(scale=0.3, size=400)
   stream[350] = 65.0

   forecaster = SNARIMAX(p=12, d=0, q=0)
   model = PredictiveAnomalyDetector(forecaster)

   for step, value in enumerate(stream):
       score = model.fit_score_partial(np.array([value]))
       if step >= 100 and score > 3.0:
           print(f"forecast error at step {step}: score={score:.2f}")

Evaluate without a holdout set
------------------------------

StatsForecast backtests a forecast on a held-out part of the series. PySAD evaluates prequentially: every point is scored before the model learns it, so there is no holdout leakage:

.. code-block:: python

   import numpy as np

   from pysad.evaluation import AUROCMetric, evaluate_stream
   from pysad.models import SeasonalHybridESD

   rng = np.random.default_rng(2)
   t = np.arange(24 * 12)
   values = 20.0 + 4.0 * np.sin(2 * np.pi * t / 24) + rng.normal(scale=0.3, size=t.size)
   labels = np.zeros(t.size, dtype=int)
   values[[220, 265]] += [6.0, -6.0]
   labels[[220, 265]] = 1

   model = SeasonalHybridESD(period=24, window_size=24 * 7, direction="both", alpha=0.001)
   metrics = evaluate_stream(
       model,
       values[:, None],
       labels,
       {"auroc": AUROCMetric()},
   )
   print({k: round(v, 3) for k, v in metrics.items() if v is not None})

Next steps
----------

- Tune alert thresholds and smoothing in :doc:`alerts`.
- Model hourly or weekly cycles explicitly in :doc:`seasonal`.
- Compose preprocessors, detectors, and calibrators in :doc:`pipelines`.
- Run pipelines behind FastAPI or a stream consumer in :doc:`serving`.
