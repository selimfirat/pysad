Univariate detectors
====================

These tables compare the univariate detectors on the series of the Numenta Anomaly Benchmark (NAB) and of the UCR Time Series Anomaly Archive :cite:`wu2021current`. The multivariate detectors also run on the NAB series, once on the single values and once on shingles, so that you can see if one of them suits a univariate stream. The best value of each column of a summary table is in bold, and :doc:`methodology` explains the steps of a run and the metrics.

The summary has one row for each detector, from the highest standard NAB score to the lowest. It has these columns:

Detector
   A row that ends with "shingles" gives a multivariate detector that gets the last 10 values as one point, from :class:`~pysad.transform.preprocessing.ShingleTransformer`.

Series
   The number of NAB series that the detector ran on. HoltWinters and the ESD detectors need the period of a series, so they run only on the 14 seasonal series.

Standard, Low false alerts, Low misses
   The NAB score with each profile, evaluated using NAB's official scoring code. Each profile optimizes a single threshold across all series. For :class:`~pysad.models.KNNCAD`, the results match the official NAB scoreboard (58.0, 43.4, and 64.8). Detectors without native unit scores in :math:`[0, 1]` are calibrated online with :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator` so their scores share a common scale across series.

AUROC
   The mean point-wise AUROC over the series with anomalies. A NAB anomaly window holds many points that look normal, so the point-wise AUROC of every detector is near 0.5. Scores are rounded to 9 significant digits before computing metrics so that identical input values produce true ties across platforms.

Time and Commit
   The median time of one ``fit_score_partial`` call in milliseconds, and the commit of PySAD that gave the results.

.. csv-table::
   :file: ../benchmark/univariate.csv
   :header-rows: 1
   :class: benchmark-table best-columns

HoltWinters and the ESD detectors run on fewer series than the other detectors, so the summary does not compare them on equal terms. The next table compares all the detectors on the 14 seasonal series only.

.. csv-table::
   :file: ../benchmark/univariate_seasonal.csv
   :header-rows: 1
   :class: benchmark-table best-columns

NAB evaluation protocol
-----------------------

The benchmark follows the evaluation protocol of the Numenta Anomaly Benchmark:

- **Official scoring algorithm**: The NAB scores use the official scoring algorithm of NAB (ported from ``sweeper.py`` and ``optimizer.py``), with window-based early-detection rewards, probationary periods (the first 15% of points, at most 750), and single global threshold optimization across all series.
- **Score calibration**: NAB's global threshold sweep requires scores across all files to share a common scale. Detectors with native unit scores in :math:`[0, 1]` (:class:`~pysad.models.HTM`, :class:`~pysad.models.KNNCAD`, and :class:`~pysad.models.RelativeEntropy`) feed raw scores directly. Other detectors are calibrated online with :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator` before thresholding.
- **Point-wise AUROC vs. NAB score**: Point-wise metrics treat every point inside a window equally and do not reward early warning or discount subsequent alerts. PySAD reports both the event-based NAB scores and point-wise AUROC for completeness.
- **Score rounding**: Series with repeated baseline values (such as idle AWS CloudWatch metrics) produce identical scores. Truncating floating-point noise to 9 significant digits ensures that identical inputs tie consistently across operating systems (Linux and macOS) rather than being arbitrarily ordered by machine-precision drift.

Each UCR series has one anomaly. The next table gives the percentage of the UCR series that each detector solves, and the mean AUROC after the training part of each series. HoltWinters and the ESD detectors do not run on these series, because the archive gives no periods.

.. csv-table::
   :file: ../benchmark/univariate_ucr.csv
   :header-rows: 1
   :class: benchmark-table best-columns
