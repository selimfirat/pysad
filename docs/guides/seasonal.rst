Detect anomalies in seasonal series
===================================

Many series of measurements repeat a cycle, for example the traffic of a website every day. Such a series is seasonal. A value that is normal at noon can be an anomaly at midnight. Seasonal detectors remove the cycle first and then look for anomalies in what is left.

Detect anomalies in an hourly series
------------------------------------

This example makes two weeks of hourly values with a daily cycle and three anomalies. :class:`~pysad.models.SeasonalHybridESD` finds them:

.. code-block:: python

   import numpy as np

   from pysad.models import SeasonalHybridESD

   rng = np.random.default_rng(0)
   t = np.arange(24 * 14)  # two weeks of hourly values
   values = 10 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(scale=0.3, size=t.size)
   values[[200, 250, 300]] += [4, -4, 5]  # three anomalies

   model = SeasonalHybridESD(period=24, window_size=24 * 7, direction="both", alpha=0.001)

   for i, value in enumerate(values):
       score = model.fit_score_partial(np.array([value]))
       if score > 0:
           print(f"anomaly at hour {i}: {value:.1f}")

.. code-block:: text

   anomaly at hour 200: 16.4
   anomaly at hour 219: 13.0
   anomaly at hour 250: 7.6
   anomaly at hour 300: 15.4

The detector tests each new value when it arrives. With ``direction="both"``, it finds the low value at hour 250 too. By default, it flags only high values, as the R package of the paper does. A fixed threshold on the raw values cannot find it, because the same value is normal at another time of day. Hour 219 is a false alert: its noise is three standard deviations above the cycle. A smaller ``alpha`` raises fewer such alerts.

How SeasonalHybridESD scores a value
------------------------------------

For each new value, the detector uses the last ``window_size`` values:

#. It removes the cycle with STL, a method that splits a series into trend, cycle and remainder. The cycle is the same in every period, so a few anomalies barely change it. Then it subtracts the median of the window.
#. It runs the generalized ESD (extreme Studentized deviate) test on the remaining values. The test removes the most extreme value again and again, up to ``max_anomalies * window_size`` times, and tests each one.
#. If the test marks the new value as an anomaly, the score is the test statistic: the distance of the value from the median, in scaled median absolute deviations (1.4826 × MAD, a robust estimate of the standard deviation). Otherwise, the score is 0.

As a result, a score above 0 is an alert. You do not need a calibrator, which maps scores to values between 0 and 1.

Set the parameters
------------------

``period``
   The number of values in one cycle. For hourly data with a daily cycle, use 24. For daily data with a weekly cycle, use 7.

``window_size``
   The number of recent values that the detector uses. It must be greater than ``2 * period``. One or two weeks of data is a good start. The detector gives 0 until the window is full.

``max_anomalies``
   The largest fraction of the window that the test looks for as anomalies, 0.1 by default. It must be at most 0.49. The test looks for at least one anomaly. A larger value finds more anomalies and raises more false alerts.

``direction``
   ``"pos"`` flags only high values, ``"neg"`` only low values, and ``"both"`` both. The default is ``"pos"``.

``alpha``
   The significance level of the test, which controls its false alert rate. A smaller value raises fewer alerts.

Choose the variant
------------------

:class:`~pysad.models.SeasonalHybridESD`
   Uses the median and the median absolute deviation in the test. Earlier anomalies in the window have little effect on it. Use this variant first.

:class:`~pysad.models.SeasonalESD`
   Uses the mean and the standard deviation. Many anomalies in the window can hide each other.

For a series without a cycle, use :class:`~pysad.models.MedianAbsoluteDeviation` or :class:`~pysad.models.StandardAbsoluteDeviation`.

A univariate detector scores points with one feature. To use another univariate detector on a seasonal series, remove the cycle first with :class:`~pysad.transform.preprocessing.SeasonalTrendDecomposer`:

.. code-block:: python

   from pysad.models import MedianAbsoluteDeviation
   from pysad.transform.preprocessing import SeasonalTrendDecomposer

   decomposer = SeasonalTrendDecomposer(season_length=24)
   model = MedianAbsoluteDeviation()

   for value in values:
       residual = decomposer.fit_transform_partial(np.array([value]))
       score = model.fit_score_partial(residual)

The decomposer subtracts a running estimate of the trend and of the cycle. It is faster than STL but less exact.

:class:`~pysad.transform.preprocessing.ModifiedSTLResidualTransformer` gives the residual that :class:`~pysad.models.SeasonalESD` tests, and it plugs into the same loop: ``ModifiedSTLResidualTransformer(period=24, window_size=24 * 14)`` in place of the decomposer. It fits STL again on the last ``window_size`` values for each new value, so it is slower.

Next steps
----------

- :doc:`alerts` explains alert thresholds for other detectors.
- :doc:`evaluate` measures the detector on labeled data.
