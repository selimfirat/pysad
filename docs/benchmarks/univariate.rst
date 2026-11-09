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
   The mean point-wise AUROC over the series with anomalies. A NAB anomaly window holds many points that look normal, so the point-wise AUROC of every detector is near 0.5. Scores are rounded to 9 significant digits before computing metrics to reduce platform-dependent floating-point noise in otherwise equal scores.

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
- **Score rounding**: Rounding scores to 9 significant digits reduces platform-dependent floating-point noise in otherwise equal scores. Repeated input values can receive different scores as a streaming detector learns, so rounding does not guarantee that identical inputs tie.

Each UCR series has one anomaly. The next table gives the percentage of the UCR series that each detector solves, and the mean AUROC after the training part of each series. HoltWinters and the ESD detectors do not run on these series, because the archive gives no periods.

.. csv-table::
   :file: ../benchmark/univariate_ucr.csv
   :header-rows: 1
   :class: benchmark-table best-columns

SAND configuration and protocol
-------------------------------

:class:`~pysad.models.SAND` :cite:`boniol2021sand` runs on all NAB series and the selected UCR series with ``subsequence_length=10`` and ``random_state=0``. The length is fixed before evaluation, using the same length as the benchmark's shingles; it uses neither anomaly labels nor the anomaly lengths encoded in UCR filenames. It is a reproducible baseline, and may miss anomalies at other scales. The centroid length remains the default, four times the subsequence length (40), and the number of clusters remains 6.

The batch size is 256 instead of the model's default of 5,000, so initialization completes within the standard warm-up and the short CI run exercises scoring. SAND returns zero for the first 256 scores, including the point that completes the first batch. Fitting that point initializes the clusters and the distance mean and standard deviation from the first batch. Each subsequent ``fit_score_partial`` call scores the subsequence ending at the current point using the last complete batch's model and normalization, then fits the point. Completing another batch updates the model only after that point has been scored. No future values or retrospective averaging of scores are used. This follows PySAD's prequential interface; the paper instead clusters a batch before scoring it, as documented in the model's docstring.

SAND uses the same evaluation cutoffs as the other detectors: point-wise NAB metrics omit the standard warm-up, NAB event scores omit NAB's probationary period, and UCR metrics omit the training prefix. These cutoffs are not extended for SAND. If a cutoff is shorter than 256 points, the remaining initialization scores count in the metrics. The first batch is not assumed to be anomaly-free. NAB event scoring uses the online conformal calibration described above; point-wise NAB and UCR metrics use the raw scores.

Run this configuration from the repository root::

   python benchmarks/benchmark.py --detectors SAND --parts univariate --jobs 1

Add ``--ci /tmp/sand.csv`` for the first 2,000 points of each NAB series, without UCR or changes to the published result tables.
