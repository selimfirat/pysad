PySAD vs Darts
==============

`Darts <https://github.com/unit8co/darts>`_ does time series forecasting. Its ``darts.ad`` module detects anomalies in a time series with four kinds of objects:

- A scorer gives an anomaly score to each time step of a series.
- A detector turns the scores into labels with a threshold.
- An aggregator combines the labels of the components of a multivariate series into one label.
- An anomaly model combines a forecasting or filtering model with one or more scorers. The scorers compare the series with the forecasts or with the filtered series.

Each of these objects works on a whole ``TimeSeries``. It learns from one series, then it scores a series. This page compares PySAD with Darts 0.48.1.

Functions
---------

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Function
     - PySAD
     - Darts
   * - Data
     - A stream, one point at a time
     - A whole ``TimeSeries``, univariate or multivariate
   * - Learn and score
     - ``fit_partial``, ``score_partial`` and ``fit_score_partial``
     - ``fit`` and ``score``, on a series
   * - Detectors on the values
     - Univariate detectors, for example :class:`~pysad.models.SeasonalESD`, :class:`~pysad.models.SeasonalHybridESD`, :class:`~pysad.models.MedianAbsoluteDeviation` and :class:`~pysad.models.SpectralResidual`, and the multivariate detectors, see :doc:`../detectors`
     - Scorers on windows of the series: ``KMeansScorer``, ``WassersteinScorer`` and ``PyODScorer``, which runs a PyOD detector
   * - Detectors on forecasts
     - :class:`~pysad.integrations.PredictiveAnomalyDetector`, with a River forecaster that learns from each value
     - ``ForecastingAnomalyModel``, with a Darts forecasting model, and ``FilteringAnomalyModel``, with a Kalman, Gaussian process or moving average filter
   * - Scores to labels
     - Probability calibrators, then a threshold on the probability
     - ``ThresholdDetector``, ``QuantileDetector`` and ``IQRDetector``
   * - Combine detectors
     - 5 score ensemblers, for example the average and the maximum of the scores
     - ``AndAggregator``, ``OrAggregator`` and ``EnsembleSklearnAggregator``, which combine labels
   * - Probability calibrators
     - Gaussian tail, conformal and SPOT
     - None
   * - Evaluation
     - ``evaluate_stream``, with AUROC, AUPR, precision and recall, also on a sliding window
     - ``eval_metric_from_scores``, with the AUROC or the AUPR of the scores of a series

The scorers of an anomaly model compare each value with its forecast. ``NormScorer`` and ``DifferenceScorer`` use the difference of the two values. The negative log-likelihood scorers, for example ``GaussianNLLScorer``, use a probabilistic forecast. :class:`~pysad.integrations.PredictiveAnomalyDetector` gives the absolute difference of the value and its forecast, or the relative difference.

The two libraries score a new value in different ways. In Darts, a model learns from a series. To score a new value, you score a series that ends with the value, and the model learns from the value only if you fit it again. In PySAD, a univariate detector learns from each value after it scores the value. For example, :class:`~pysad.models.SeasonalESD` keeps a window of recent values and removes the seasonal part of the window before it scores the newest value.

Use Darts if you have the whole series, for example to find anomalies in past data, or if you already forecast with Darts. Use PySAD if the values arrive one at a time and each value needs a score when it arrives.

This page has no benchmark results, because Darts does not learn from one value at a time.
