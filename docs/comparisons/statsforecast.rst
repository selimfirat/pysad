.. meta::
   :description: PySAD vs StatsForecast/Nixtla for time series anomaly detection: compare point-by-point streaming detectors with high-performance batch forecasting models.

PySAD vs StatsForecast / Nixtla
===============================

`StatsForecast <https://github.com/Nixtla/statsforecast>`_ is a high-performance time series forecasting library from Nixtla. It provides fast implementations of statistical models such as AutoARIMA, AutoETS, MSTL, Theta, and Croston. It forecasts a whole series or a batch of series. It does not learn from one point at a time. This page compares PySAD with StatsForecast 1.7.0.

Functions
---------

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Function
     - PySAD
     - StatsForecast
   * - Data
     - A stream, one point at a time
     - A whole ``pandas.DataFrame`` (long format) or a batch of series
   * - Learn and score
     - ``fit_partial``, ``score_partial`` and ``fit_score_partial``
     - ``fit`` and ``predict`` on a batch of series
   * - Detectors on the values
     - Univariate detectors, for example :class:`~pysad.models.SeasonalESD`, :class:`~pysad.models.SeasonalHybridESD`, :class:`~pysad.models.MedianAbsoluteDeviation` and :class:`~pysad.models.SpectralResidual`, and the multivariate detectors, see :doc:`../detectors`
     - None (forecasting models only)
   * - Detectors on forecasts
     - :class:`~pysad.integrations.PredictiveAnomalyDetector`, with a River forecaster that learns from each value
     - Forecast with AutoARIMA, AutoETS, MSTL, Theta, Croston, and others; then compare the forecast with the observed value
   * - Scores to labels
     - Probability calibrators, then a threshold on the probability
     - None (user applies a threshold to the forecast error)
   * - Combine detectors
     - 5 score ensemblers, for example the average and the maximum of the scores
     - None
   * - Probability calibrators
     - Gaussian tail, conformal and SPOT
     - None
   * - Evaluation
     - ``evaluate_stream``, with AUROC, AUPR, precision and recall, also on a sliding window
     - Cross-validation and backtesting utilities for forecasting accuracy

StatsForecast's models learn from a whole training series and then forecast future steps. To detect anomalies, you compare the forecast with the observed value and flag large errors. In PySAD, a univariate detector learns from each value after it scores the value. For example, :class:`~pysad.models.SeasonalESD` keeps a window of recent values and removes the seasonal part of the window before it scores the newest value. :class:`~pysad.integrations.PredictiveAnomalyDetector` wraps a River forecaster that learns from each value, giving an anomaly score for each new point without refitting.

Use StatsForecast if you have the whole series and need fast batch forecasting with statistical models. Use PySAD if the values arrive one at a time and each value needs a score when it arrives, or if you need probability calibrators, score ensemblers, and streaming evaluation.

This page has no benchmark results, because StatsForecast does not learn from one value at a time.
