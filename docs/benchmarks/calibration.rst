Calibrators
===========

A calibrator turns detector scores into values from 0 to 1. A calibrated score at or above a threshold raises an alert. See :doc:`/guides/alerts` for examples. This table compares the two probability calibrators on multivariate detector runs, with their default parameters.

The fixed thresholds are 0.95, 0.99 and 0.995. The threshold "1 - anomaly rate" uses the labels, which a live stream does not provide. The highest precision, recall, F1 and MCC are in bold. See :doc:`methodology` for the alert rules. The table has these columns:

Expected
   The expected false alert rate, one minus the threshold.

False alerts
   The fraction of normal points that raised an alert, as a mean over all the detectors and datasets.

Range
   The lowest and the highest mean false alert rate of a detector. A small range means that the threshold has the same effect for every detector.

Precision, Recall and F1
   The fraction of the alerts that are anomalies, the fraction of the anomalies that raised an alert, and their harmonic mean. A run without alerts does not count in the precision, and the cell is empty when no run had alerts.

MCC
   The mean Matthews correlation coefficient over all valid runs, at the same thresholds as precision and recall.
   It counts true and false predictions for both classes.
   A score of 1 means perfect agreement. A score of -1 means complete disagreement.
   Constant predictions give 0. Invalid runs do not count.
   An empty cell means that no run has an MCC value. Rerun the benchmark to fill it.

.. csv-table::
   :file: ../benchmark/alerts.csv
   :header-rows: 1
   :class: benchmark-table best-columns
