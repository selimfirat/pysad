Univariate detectors
====================

This page compares the univariate detectors of PySAD on the series of the Numenta Anomaly Benchmark (NAB), with the score of NAB. A univariate detector takes one value for each point. Each detector uses its default parameters, plus the arguments that it requires. Use the results to find two or three candidates. Then compare the candidates on your own data, as :doc:`/guides/choose_detector` shows.

The NAB score
-------------

NAB marks a window of points around each anomaly. Its score counts each window as one event:

- An alert in a window detects it. An early alert counts more than a late one, and only the best alert of a window counts.
- A window without an alert is a miss.
- An alert outside the windows is a false alert. A false alert soon after a window costs less than one far from all windows.

NAB uses one alert threshold for all the series of a detector, and it chooses the threshold that gives the highest score. Then it scales the score: a detector that raises no alert gets 0, and a detector that alerts at the start of every window gets 100. A detector with many false alerts can get less than 0.

NAB has three profiles, which weight the false alerts and the misses in different ways. The standard profile weights a false alert at 0.11 of a miss. The profile "low false alerts" doubles the weight of a false alert, and the profile "low misses" doubles the weight of a miss.

Results
-------

Summary
~~~~~~~

The summary has one row for each detector, from the highest standard NAB score to the lowest. It has these columns:

Series
   The number of NAB series that the detector ran on. The ESD detectors need the period of a series, so they run only on the seasonal series.

NAB standard, NAB low false alerts, NAB low misses
   The NAB score with each profile.

Mean AUROC
   The mean AUROC over the series, with each point of a window as an anomaly. NAB windows hold many points that look normal, so the AUROC of every detector is near 0.5. The NAB score is the better measure for these series.

Time per point
   The median time of one ``fit_score_partial`` call, in milliseconds.

Commit
   The commit of PySAD that gave the results.

.. csv-table::
   :file: ../benchmark/univariate.csv
   :header-rows: 1
   :class: benchmark-table

Seasonal series
~~~~~~~~~~~~~~~

The ESD detectors run on fewer series than the others, so the summary does not compare them on equal terms. This table compares all the detectors on the 5 seasonal series only, with the standard profile.

.. csv-table::
   :file: ../benchmark/univariate_seasonal.csv
   :header-rows: 1
   :class: benchmark-table

How the benchmark works
-----------------------

NAB has 58 series, and 52 of them have anomalies. For each of the 52 series and each detector, the benchmark does these steps:

#. It loads the series with :class:`~pysad.utils.Data` in its time order, for example with ``Data().get_data("nab/realKnownCause/nyc_taxi.csv")``. Each point in a window has the label 1.
#. It scales the values to the range from 0 to 1, with the minimum and the maximum of the series.
#. It makes the detector with its default parameters. Some detectors also get the arguments that they require. :class:`~pysad.models.KNNCAD` learns during the probationary period of NAB, as in NAB. :class:`~pysad.models.RelativeEntropy` gets the range from 0 to 1. The ESD detectors get the period of the series, a window of 3 periods and at most 10% of the window as anomalies, which is the default of the R package of their paper.
#. It gives each point to ``fit_score_partial``.
#. NAB uses one threshold for all the series of a detector, so the scores must have the same scale on each series. :class:`~pysad.models.KNNCAD` and :class:`~pysad.models.RelativeEntropy` give scores from 0 to 1, as the detectors of NAB do. The benchmark calibrates the scores of the other detectors with :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator`, which gives values from 0 to 1 on every series.
#. It computes the NAB score of the detector over all its series, with the scoring code of NAB. As in NAB, the score leaves out the probationary period of each series: the first 15% of its points, at most 750.

With these steps, :class:`~pysad.models.KNNCAD` gets NAB scores of 58.4, 44.3 and 65.1. The scoreboard of NAB lists 58.0, 43.4 and 64.8 for its KNN-CAD detector, so the benchmark reproduces the method of NAB.

The mean AUROC leaves out the first 256 points of each series, as the :doc:`methodology` of the multivariate benchmark does. None of these detectors draws random numbers, so each one runs once on each series, with no seeds.

The period of a seasonal series is one day, in points. A series counts as seasonal when the correlation of its values with the values one day earlier is at least 0.5. The ESD detectors run on the 5 seasonal series with a period of 24 or 48 points. Nine more series have a daily period of 288 points. A window of 3 such periods makes the STL decomposition of each point too slow for this benchmark, so the ESD detectors do not run on them.

What the results do not show
----------------------------

- The NAB scoreboard has no results for MAD, SAD and the ESD detectors. Their scores here depend on the conformal calibration, which NAB does not use.
- The ESD detectors differ from the R package of their paper in some settings, for example in their STL settings and in the direction of the test.
- The periods come from the data, not from NAB. NAB gives no periods.
