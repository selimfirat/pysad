.. meta::
   :description: PySAD vs ADTK for time series anomaly detection: compare point-by-point streaming detectors with rule-based and statistical batch methods.

PySAD vs ADTK
=============

`ADTK <https://github.com/arundo/adtk>`_ is a Python toolkit for unsupervised and rule-based anomaly detection on time series. It applies threshold, volatility, seasonal, and level-shift detectors to a whole series or a rolling window. It does not learn from one point at a time. This page compares PySAD with ADTK 0.14.1.

Functions
---------

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Function
     - PySAD
     - ADTK
   * - Data
     - A stream, one point at a time
     - A whole ``pandas.Series`` or ``pandas.DataFrame``, or a rolling window
   * - Learn and score
     - ``fit_partial``, ``score_partial`` and ``fit_score_partial``
     - ``fit`` and ``detect`` on a series, or ``transform`` on a rolling window
   * - Detectors on the values
     - Univariate detectors, for example :class:`~pysad.models.SeasonalESD`, :class:`~pysad.models.SeasonalHybridESD`, :class:`~pysad.models.MedianAbsoluteDeviation` and :class:`~pysad.models.SpectralResidual`, and the multivariate detectors, see :doc:`../detectors`
     - Rule-based and statistical: ``ThresholdAD``, ``VolatilityAD``, ``InterQuartileRangeAD``, ``GeneralizedESDTestAD``, ``PersistAD``, ``SeasonalAD``, ``LevelShiftAD`` and ``CustomRuleAD``
   * - Detectors on forecasts
     - :class:`~pysad.integrations.PredictiveAnomalyDetector`, with a River forecaster that learns from each value
     - None
   * - Scores to labels
     - Probability calibrators, then a threshold on the probability
     - Built into each detector, for example a threshold, a quantile, or the IQR rule
   * - Combine detectors
     - 5 score ensemblers, for example the average and the maximum of the scores
     - ``AndAggregation``, ``OrAggregation`` and ``MajorityVoteAggregation`` on labels
   * - Probability calibrators
     - Gaussian tail, conformal and SPOT
     - None
   * - Evaluation
     - ``evaluate_stream``, with AUROC, AUPR, precision and recall, also on a sliding window
     - None

ADTK's detectors apply a rule or a statistical test to a window of values. For example, ``SeasonalAD`` decomposes the series and flags values where the residual exceeds a multiple of the standard deviation. ``LevelShiftAD`` uses a change-point test on the mean. In PySAD, a univariate detector learns from each value after it scores the value. For example, :class:`~pysad.models.SeasonalESD` keeps a window of recent values and removes the seasonal part of the window before it scores the newest value.

Use ADTK if you have the whole series and want rule-based or statistical detectors that do not need training. Use PySAD if the values arrive one at a time and each value needs a score when it arrives, or if you need probability calibrators and score ensemblers.

This page has no benchmark results, because ADTK does not learn from one value at a time.