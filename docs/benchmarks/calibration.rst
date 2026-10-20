Calibrators
===========

A calibrator turns the scores of a detector into values from 0 to 1, and an alert is a calibrated score at or above a threshold, as :doc:`/guides/alerts` shows. This table compares the two probability calibrators of PySAD on the runs of the multivariate detectors, with their default parameters. The fixed thresholds 0.95, 0.99 and 0.995 give the fraction of normal points that raise an alert. The threshold "1 - anomaly rate" gives as many alerts as there are anomalies. It uses the labels, so it shows the best case. The highest precision, recall and F1 are in bold, and :doc:`methodology` explains how the benchmark counts the alerts. The table has these columns:

Expected
   The expected false alert rate, one minus the threshold.

False alerts
   The fraction of normal points that raised an alert, as a mean over all the detectors and datasets.

Range
   The lowest and the highest mean false alert rate of a detector. A small range means that the threshold has the same effect for every detector.

Precision, Recall and F1
   The fraction of the alerts that are anomalies, the fraction of the anomalies that raised an alert, and their harmonic mean. A run without alerts does not count in the precision, and the cell is empty when no run had alerts.

.. csv-table::
   :file: ../benchmark/alerts.csv
   :header-rows: 1
   :class: benchmark-table best-columns
