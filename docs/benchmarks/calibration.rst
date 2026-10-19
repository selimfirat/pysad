Calibrators
===========

This page compares the two probability calibrators of PySAD on the same runs as the :doc:`multivariate detector benchmark </benchmark>`. A calibrator turns the scores of a detector into values from 0 to 1. An alert is a calibrated score at or above a threshold, as :doc:`/guides/alerts` shows. For each run, the benchmark gives the scores to each calibrator with its default parameters. Then it counts the alerts after the warm-up, which :doc:`methodology` explains. The calibrators also learn from the warm-up points.

Thresholds
----------

The benchmark uses two kinds of threshold. The fixed thresholds 0.95, 0.99 and 0.995 test what the alerts guide gives: the fraction of normal points that raise an alert. This fraction does not depend on the number of anomalies in a dataset. Precision and recall at a fixed threshold do depend on it. For example, a threshold of 0.99 can find only a small part of the anomalies of pima, where 35% of the points are anomalies.

The threshold "1 - anomaly rate" is different for each run, so that the number of alerts is close to the number of anomalies. It compares the calibrators on equal terms over all the datasets. But it uses the labels, which a live stream does not have, so it shows the best case.

Results
-------

The table has these columns:

Expected
   The expected false alert rate, one minus the threshold. The alerts guide gives this value as the fraction of normal points that raise an alert. The threshold "1 - anomaly rate" has no fixed value.

False alerts
   The fraction of normal points that raised an alert. It is the mean over all the detectors, datasets and seeds.

Range
   The lowest and the highest mean false alert rate of a detector. A small range means that the threshold has the same effect for every detector.

Precision
   The fraction of the alerts that are anomalies. A run without alerts does not count.

Recall
   The fraction of the anomalies that raised an alert.

F1
   The harmonic mean of the precision and the recall. A run without alerts counts as 0.

.. csv-table::
   :file: ../benchmark/alerts.csv
   :header-rows: 1
   :class: benchmark-table

The anomalies also enter the window of a calibrator. On a dataset with many anomalies, fewer normal points pass the threshold, so the false alert rate can be lower than expected. The window of :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator` holds 300 scores by default. Its highest calibrated score is thus 299/300, about 0.997, so a threshold of 0.999 never raises an alert with the defaults.
