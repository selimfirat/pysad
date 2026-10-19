Univariate detectors
====================

This page compares the univariate detectors of PySAD on the series of the Numenta Anomaly Benchmark (NAB). A univariate detector takes one value for each point. Each detector uses its default parameters, plus the arguments that it requires. Use the results to find two or three candidates. Then compare the candidates on your own data, as :doc:`/guides/choose_detector` shows.

Results
-------

Summary
~~~~~~~

The summary has one row for each detector, from the highest mean AUROC to the lowest. :doc:`/benchmark` explains AUROC and AUPR. It has these columns:

Series
   The number of NAB series that the detector ran on. The ESD detectors need the period of a series, so they run only on the seasonal series.

Mean AUROC and Mean AUPR
   The averages over these series.

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

The ESD detectors run on fewer series than the others, so the summary does not compare them on equal terms. This table compares all the detectors on the 5 seasonal series only.

.. csv-table::
   :file: ../benchmark/univariate_seasonal.csv
   :header-rows: 1
   :class: benchmark-table

AUROC on each series
~~~~~~~~~~~~~~~~~~~~

The highest value on each series is in bold. An empty cell means that the detector did not run on the series. The columns are in the order of the summary. The table uses these short names: MAD is :class:`~pysad.models.MedianAbsoluteDeviation`, RE is :class:`~pysad.models.RelativeEntropy`, SAD is :class:`~pysad.models.StandardAbsoluteDeviation`, S-ESD is :class:`~pysad.models.SeasonalESD`, and S-H-ESD is :class:`~pysad.models.SeasonalHybridESD`.

.. csv-table::
   :file: ../benchmark/univariate_auroc.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

How the benchmark works
-----------------------

NAB has 58 series, and 52 of them have anomalies. NAB marks a window of points around each anomaly, and the benchmark labels every point in a window as an anomaly. For each of the 52 series and each detector, the benchmark does these steps:

#. It loads the series with :class:`~pysad.utils.Data` in its time order, for example with ``Data().get_data("nab/realKnownCause/nyc_taxi.csv")``.
#. It scales the values to the range from 0 to 1, with the minimum and the maximum of the series.
#. It makes the detector with its default parameters. Some detectors also get the arguments that they require. :class:`~pysad.models.KNNCAD` learns during a probationary period of 256 points, the warm-up. :class:`~pysad.models.RelativeEntropy` gets the range from 0 to 1. The ESD detectors get the period of the series, a window of 3 periods and at most 10% of the window as anomalies, which is the default of the R package of their paper.
#. It gives each point to ``fit_score_partial``.
#. It computes AUROC and AUPR over the scores of the points after the warm-up. The warm-up is the first 256 points, as in the :doc:`methodology` of the multivariate benchmark. Every NAB series has more than 1,024 points.

None of these detectors draws random numbers, so each one runs once on each series, with no seeds.

The period of a seasonal series is one day, in points. A series counts as seasonal when the correlation of its values with the values one day earlier is at least 0.5. The ESD detectors run on the 5 seasonal series with a period of 24 or 48 points. Nine more series have a daily period of 288 points. A window of 3 such periods makes the STL decomposition of each point too slow for this benchmark, so the ESD detectors do not run on them.

What the results do not show
----------------------------

- NAB ranks detectors with its own score, which rewards an early detection in each window. This benchmark uses AUROC and AUPR over the points instead, as the multivariate benchmark does. So the results are not the NAB scores of the detectors.
- The ESD detectors differ from the R package of their paper in some settings, for example in their STL settings and in the direction of the test.
- The periods come from the data, not from NAB. NAB gives no periods.
