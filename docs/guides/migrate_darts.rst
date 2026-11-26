.. meta::
   :description: Migrate time series anomaly detection from unit8co Darts (0.48.1) to PySAD streaming pipelines on Python 3.11+ with NumPy 2.

Migrate from unit8co Darts to PySAD
=====================================

`Darts <https://github.com/unit8co/darts>`_ is a time series forecasting library. Its ``darts.ad`` module detects anomalies in a whole ``TimeSeries``. Each object in ``darts.ad``, scorer, detector, aggregator, and anomaly model, learns from one series and then scores a series. It does not learn from one point at a time. This page compares PySAD with Darts 0.48.1 and shows how to move a Darts anomaly workflow to PySAD.

PySAD scores each incoming point before it learns from it, keeps memory bounded, and needs only ``numpy``, ``scipy``, and ``scikit-learn`` on Python 3.11+. No new dependency is required to move a Darts workflow to PySAD.

Scope and architectural differences
-----------------------------------

- **Batch series versus streaming points**: Darts fits on a complete ``TimeSeries`` and scores a series. PySAD scores each point with ``fit_score_partial`` and then updates its state, so every point gets a score when it arrives.
- **No streaming state**: Darts scorers and anomaly models refit on the whole series to learn from new values. PySAD detectors keep a bounded window or summary instead of the full history.
- **Input format**: Darts takes a univariate or multivariate ``TimeSeries``. PySAD takes one NumPy vector of shape ``(num_features,)`` per point. Use :class:`~pysad.utils.PandasStreamer` to stream a DataFrame row by row.
- **Keep Darts for batch jobs**: If you already run Darts forecasts on a complete series overnight, keep that job. Move to PySAD when values arrive one at a time or when you need calibrated streaming alerts.

API mapping
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Darts component
     - PySAD equivalent
     - Notes
   * - A whole ``TimeSeries``
     - One NumPy vector of shape ``(num_features,)`` per point
     - Stream a pandas DataFrame with :class:`~pysad.utils.PandasStreamer`.
   * - ``fit`` then ``score`` on a series
     - ``fit_score_partial``, then ``fit_partial`` and ``score_partial``
     - Score each point before the model learns it.
   * - ``KMeansScorer``, ``WassersteinScorer``, ``PyODScorer``
     - :class:`~pysad.models.SeasonalESD`, :class:`~pysad.models.SeasonalHybridESD`, :class:`~pysad.models.MedianAbsoluteDeviation`, :class:`~pysad.models.SpectralResidual`
     - Streaming detectors that score each value as it arrives.
   * - ``ForecastingAnomalyModel``, ``FilteringAnomalyModel``
     - :class:`~pysad.integrations.PredictiveAnomalyDetector` with a River forecaster
     - Wraps a River forecaster such as SNARIMAX. Install River with ``pip install river``.
   * - ``ThresholdDetector``, ``QuantileDetector``, ``IQRDetector``
     - :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator`, :class:`~pysad.transform.probability_calibration.SPOTProbabilityCalibrator`, :class:`~pysad.transform.postprocessing.ThresholdFilter`
     - Map scores to probabilities in ``[0, 1]``, then apply a fixed cutoff.
   * - ``AndAggregator``, ``OrAggregator``, ``EnsembleSklearnAggregator``
     - :class:`~pysad.models.ModelEnsemble` or the ``&`` operator, :mod:`pysad.transform.ensemble`
     - Combines detector scores or pipelines.
   * - ``eval_metric_from_scores``
     - :func:`~pysad.evaluation.evaluate_stream` with :class:`~pysad.evaluation.AUROCMetric`
     - Computes AUROC and AUPR prequentially: each point is scored before the model learns it.

Replace forecast-based anomaly detection on a stream
----------------------------------------------------

Darts users often fit a forecasting model on history, forecast the next value, and flag points whose error exceeds a bound. On a stream, replace that loop with a streaming detector and a calibrator. This example uses only PySAD and NumPy:

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

Darts evaluates scores on a series with ``eval_metric_from_scores``. PySAD evaluates prequentially: every point is scored before the model learns it, so there is no holdout leakage:

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
