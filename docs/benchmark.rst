Benchmark results
=================

The benchmark runs the detectors of PySAD on labeled data and compares their results. Each detector runs once on each dataset, with its default parameters and seed 0. Use the results to find two or three candidates for your data. Then compare these candidates on your own data, as :doc:`guides/choose_detector` shows.

A multivariate detector takes a vector of features for each point. A univariate detector takes one value for each point. :doc:`benchmarks/methodology` explains the steps of a run, the metrics, the datasets and the limits of the results. In each summary table, the best value of each column is in bold.

Multivariate detectors
----------------------

These tables compare the 10 multivariate detectors on |num_datasets| datasets from the `ODDS <http://odds.cs.stonybrook.edu/>`_ repository. The points of each dataset are in a random order. The summary has one row for each detector, from the best mean rank to the worst. It has these columns:

Mean rank
   The rank of a detector on a dataset is its position by AUROC, where 1 is the highest AUROC. The mean rank is the average of these ranks over the |num_datasets| datasets. Detectors with the same mean rank are in the order of their mean AUROC, then their mean AUPR, then their time per point.

Mean AUROC and Mean AUPR
   The averages over the |num_datasets| datasets.

Time per point
   The median time of one ``fit_score_partial`` call, in milliseconds. All the runs used one computer, so compare these times only with each other.

Commit
   The commit of PySAD that gave the results.

.. csv-table::
   :file: benchmark/summary.csv
   :header-rows: 1
   :class: benchmark-table best-columns

The next table gives the AUROC on each dataset. The highest value on each dataset is in bold, and the columns are in the order of the summary. HST is :class:`~pysad.models.HalfSpaceTrees`, LoOP is :class:`~pysad.models.LocalOutlierProbability`, and RRCF is :class:`~pysad.models.RobustRandomCutForest`.

.. csv-table::
   :file: benchmark/auroc.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

The next table gives the AUPR on each dataset. The AUPR of a random detector is the fraction of anomalies, which is different on each dataset. Thus, compare the values only in the same row.

.. csv-table::
   :file: benchmark/aupr.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

Streams with drift
------------------

In a real stream, the normal points change over time, and an old pattern can come back. This change is called drift. Many detectors forget old points to follow drift, for example with a sliding window. These tables compare the 10 multivariate detectors on 8 real streams in their time order:

- KDD http: network connections, with attacks in 10 bursts.
- INSECTS A, I, IG and IR: optical sensor values of flying insects, with abrupt, incremental, incremental and gradual, and incremental and reoccurring drift.
- SMD 1-1, 2-1 and 3-1: server machines, with one point of 38 metrics each minute.

The tables have the same columns as the tables of the multivariate detectors.

.. csv-table::
   :file: benchmark/drift_summary.csv
   :header-rows: 1
   :class: benchmark-table best-columns

.. csv-table::
   :file: benchmark/drift_auroc.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

.. csv-table::
   :file: benchmark/drift_aupr.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

Univariate detectors
--------------------

These tables compare the univariate detectors on the series of the Numenta Anomaly Benchmark (NAB) and of the UCR Time Series Anomaly Archive :cite:`wu2021current`. The multivariate detectors also run on the NAB series, once on the single values and once on shingles, so that you can see if one of them suits a univariate stream. The summary has one row for each detector, from the highest standard NAB score to the lowest. It has these columns:

Detector
   A row that ends with "shingles" gives a multivariate detector that gets the last 10 values as one point, from :class:`~pysad.transform.preprocessing.ShingleTransformer`.

Series
   The number of NAB series that the detector ran on. The ESD detectors need the period of a series, so they run only on the 14 seasonal series.

Standard, Low false alerts, Low misses
   The NAB score with each profile.

AUROC
   The mean AUROC over the series with anomalies. A NAB window holds many points that look normal, so the AUROC of every detector is near 0.5.

Time and Commit
   The median time of one ``fit_score_partial`` call in milliseconds, and the commit of PySAD that gave the results.

.. csv-table::
   :file: benchmark/univariate.csv
   :header-rows: 1
   :class: benchmark-table best-columns

The ESD detectors run on fewer series than the other detectors, so the summary does not compare them on equal terms. The next table compares all the detectors on the 14 seasonal series only.

.. csv-table::
   :file: benchmark/univariate_seasonal.csv
   :header-rows: 1
   :class: benchmark-table best-columns

Each UCR series has one anomaly. The next table gives the percentage of the UCR series that each detector solves, and the mean AUROC after the training part of each series. The ESD detectors do not run on these series, because the archive gives no periods.

.. csv-table::
   :file: benchmark/univariate_ucr.csv
   :header-rows: 1
   :class: benchmark-table best-columns

Calibrators
-----------

A calibrator turns the scores of a detector into values from 0 to 1, and an alert is a calibrated score at or above a threshold, as :doc:`guides/alerts` shows. This table compares the two probability calibrators of PySAD on the runs of the multivariate detectors, with their default parameters. The fixed thresholds 0.95, 0.99 and 0.995 give the fraction of normal points that raise an alert. The threshold "1 - anomaly rate" gives as many alerts as there are anomalies. It uses the labels, so it shows the best case. The table has these columns:

Expected
   The expected false alert rate, one minus the threshold.

False alerts
   The fraction of normal points that raised an alert, as a mean over all the detectors and datasets.

Range
   The lowest and the highest mean false alert rate of a detector. A small range means that the threshold has the same effect for every detector.

Precision, Recall and F1
   The fraction of the alerts that are anomalies, the fraction of the anomalies that raised an alert, and their harmonic mean. A run without alerts does not count in the precision, and the cell is empty when no run had alerts.

.. csv-table::
   :file: benchmark/alerts.csv
   :header-rows: 1
   :class: benchmark-table best-columns
