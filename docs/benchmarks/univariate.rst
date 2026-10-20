Univariate detectors
====================

This page compares the univariate detectors on the series of the Numenta Anomaly Benchmark (NAB) and of the UCR Time Series Anomaly Archive :cite:`wu2021current`. The multivariate detectors also run on the NAB series, once on the single values and once on shingles. Thus, you can see if a multivariate detector suits a univariate stream. :doc:`methodology` explains the NAB score and the steps of a run.

NAB series
----------

The summary has one row for each detector, from the highest standard NAB score to the lowest. The best value of each score and of the time is in bold. The summary has these columns:

Detector
   A row that ends with "shingles" gives a multivariate detector that gets the last 10 values as one point. :class:`~pysad.transform.preprocessing.ShingleTransformer` makes these points.

Series
   The number of NAB series that the detector ran on. The ESD detectors need the period of a series, so they run only on the 14 seasonal series.

Standard, Low false alerts, Low misses
   The NAB score with each profile.

AUROC
   The mean AUROC over the series with anomalies. A NAB window holds many points that look normal, so the AUROC of every detector is near 0.5.

Time and Commit
   The median time of one ``fit_score_partial`` call in milliseconds, and the commit of PySAD that gave the results.

.. csv-table::
   :file: ../benchmark/univariate.csv
   :header-rows: 1
   :class: benchmark-table best-columns

Seasonal series
---------------

The ESD detectors run on fewer series than the other detectors, so the summary does not compare them on equal terms. This table compares all the detectors on the 14 seasonal series only.

.. csv-table::
   :file: ../benchmark/univariate_seasonal.csv
   :header-rows: 1
   :class: benchmark-table best-columns

UCR series
----------

Each UCR series has one anomaly. The table gives the percentage of the series that each detector solves, and the mean AUROC after the training part of each series. The ESD detectors do not run on these series, because the archive gives no periods.

.. csv-table::
   :file: ../benchmark/univariate_ucr.csv
   :header-rows: 1
   :class: benchmark-table best-columns
