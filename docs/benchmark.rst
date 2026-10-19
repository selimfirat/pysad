Benchmark
=========

This page compares the multivariate detectors and the calibrators of PySAD on 17 labeled datasets. A multivariate detector takes a vector of features for each point. Each detector uses its default parameters. Use the results to find two or three candidates. Then compare the candidates on your own data, as :doc:`guides/choose_detector` shows.

Results
-------

The benchmark measures how well the scores of a detector separate the anomalies from the normal points. It uses two metrics:

AUROC
   The probability that a random anomaly gets a higher score than a random normal point. A random detector gets 0.5. A perfect detector gets 1.

AUPR
   The average precision over all the score thresholds. A random detector gets the fraction of anomalies in the dataset. A perfect detector gets 1.

:doc:`guides/evaluate` explains the two metrics in more detail.

Summary
~~~~~~~

The summary has one row for each detector, from the best mean rank to the worst. Detectors with the same mean rank are in the order of their mean AUROC, then their mean AUPR, then their time per point. It has these columns:

Mean rank
   The rank of a detector on a dataset is its position by AUROC, where 1 is the highest AUROC. The mean rank is the average of these ranks over the 17 datasets.

Mean AUROC and Mean AUPR
   The averages over the 17 datasets and the 5 seeds.

Time per point
   The median time of one ``fit_score_partial`` call, in milliseconds. The runs shared one machine, so compare these times only with each other.

Commit
   The commit of PySAD that gave the results. Click it to see the code on GitHub.

.. csv-table::
   :file: benchmark/summary.csv
   :header-rows: 1
   :class: benchmark-table

AUROC on each dataset
~~~~~~~~~~~~~~~~~~~~~

Each value is the mean over the 5 seeds. The highest value on each dataset is in bold. The columns are in the order of the summary. The table uses three short names: HST is :class:`~pysad.models.HalfSpaceTrees`, LoOP is :class:`~pysad.models.LocalOutlierProbability`, and RRCF is :class:`~pysad.models.RobustRandomCutForest`.

.. csv-table::
   :file: benchmark/auroc.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

AUPR on each dataset
~~~~~~~~~~~~~~~~~~~~

This table has the same layout as the AUROC table. The AUPR of a random detector is different on each dataset, so compare the values only in the same row.

.. csv-table::
   :file: benchmark/aupr.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

Alerts after calibration
~~~~~~~~~~~~~~~~~~~~~~~~

A calibrator turns the scores of a detector into values from 0 to 1. An alert is a calibrated score at or above a threshold, as :doc:`guides/alerts` shows. For each run, the benchmark gives the scores to each calibrator with its default parameters. Then it counts the alerts after the warm-up.

The benchmark uses two kinds of threshold. The fixed thresholds 0.95, 0.99 and 0.995 test what the alerts guide gives: the fraction of normal points that raise an alert. This fraction does not depend on the number of anomalies in a dataset. Precision and recall at a fixed threshold do depend on it. For example, a threshold of 0.99 can find only a small part of the anomalies of pima, where 35% of the points are anomalies. The threshold "1 - anomaly rate" is different for each run, so that the number of alerts is close to the number of anomalies. It compares the calibrators on equal terms over all the datasets. But it uses the labels, which a live stream does not have, so it shows the best case.

The table has these columns:

Expected false alerts
   One minus the threshold. The alerts guide gives this value as the fraction of normal points that raise an alert. The threshold "1 - anomaly rate" has no fixed value.

False alerts
   The fraction of normal points that raised an alert. It is the mean over all the detectors, datasets and seeds.

Range over detectors
   The lowest and the highest mean false alert rate of a detector. A small range means that the threshold has the same effect for every detector.

Precision
   The fraction of the alerts that are anomalies. A run without alerts does not count.

Recall
   The fraction of the anomalies that raised an alert.

F1
   The harmonic mean of the precision and the recall. A run without alerts counts as 0.

.. csv-table::
   :file: benchmark/alerts.csv
   :header-rows: 1
   :class: benchmark-table

The anomalies also enter the window of a calibrator. On a dataset with many anomalies, fewer normal points pass the threshold, so the false alert rate can be lower than expected. The window of :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator` holds 300 scores by default. Its highest calibrated score is thus 299/300, about 0.997, so a threshold of 0.999 never raises an alert with the defaults.

How the benchmark works
-----------------------

The benchmark follows the rules of :doc:`guides/evaluate`. For each dataset, each detector and each seed from 0 to 4, it does these steps:

#. It loads the dataset with :class:`~pysad.utils.Data`. Then it shuffles the points with ``ArrayStreamer(shuffle=True, random_state=seed)``. Many of the files put all the anomalies at the end, so the order of the file is not a fair stream.
#. It keeps the first 5,000 points of the shuffled stream. The first 256 of these points are the warm-up. A stream with fewer than 1,024 points has a warm-up of a quarter of its points.
#. It scales each feature to the range from 0 to 1, with the minimum and the maximum from :func:`~pysad.utils.get_minmax_array`. HST and :class:`~pysad.models.RSHash` require the range of each feature in advance, and after the scaling, this range is from 0 to 1.
#. It makes the detector with its default parameters and ``random_state=seed``. Some detectors also get the arguments that they require. HST and :class:`~pysad.models.RSHash` get the range from 0 to 1 for each feature. :class:`~pysad.models.Inqmad` gets the number of features. LoOP fits on the warm-up points.
#. It gives each point to ``fit_score_partial``, which scores the point and then trains the detector on it.
#. It computes AUROC and AUPR over the scores of the points after the warm-up. The detector learns from the warm-up points, but it has little or no model when it scores them. The xStream paper and the Numenta Anomaly Benchmark leave out such points in the same way.

Step 3 uses the minimum and the maximum of the whole stream. A live stream does not know them in advance. For a live stream, take them from historical data, or use known physical limits.

Datasets
~~~~~~~~

The datasets come from the ODDS repository (Outlier Detection DataSets). The column "Points used" gives the number of points after the limit of 5,000. The column "Points evaluated" gives the number of points after the warm-up.

.. csv-table::
   :file: benchmark/datasets.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

What the results do not show
----------------------------

- The univariate detectors take one value per point, so the benchmark does not run them. These detectors are :class:`~pysad.models.MedianAbsoluteDeviation`, :class:`~pysad.models.StandardAbsoluteDeviation`, :class:`~pysad.models.KNNCAD`, :class:`~pysad.models.RelativeEntropy`, :class:`~pysad.models.SeasonalESD` and :class:`~pysad.models.SeasonalHybridESD`.
- :class:`~pysad.models.integrations.ReferenceWindowModel` runs a PyOD model, so its results depend on that model. :class:`~pysad.models.IForestASD` is one example of it.
- The points of these datasets are independent, and the benchmark puts them in a random order. The streams have no drift and no seasonality. Thus the results do not show how a detector follows changes over time.
- Tuned parameters can give very different results. For example, :class:`~pysad.models.KitNet` trains on its first 1,100 points by default. 7 of these datasets have fewer points, so on them, KitNet never ends its training. :doc:`guides/choose_detector` names the parameters with the largest effect.

Run the benchmark
-----------------

Install PySAD in editable mode, as :doc:`contributing` shows. Then run the benchmark from the root of the repository:

.. code-block:: console

   $ python benchmarks/benchmark.py

The first run downloads the datasets. The script writes the results to ``docs/benchmark/``, and this page shows them. A full run takes about 2.5 CPU-hours, which is 21 minutes on 10 cores. The ``--jobs`` option sets the number of processes. The default is the number of CPU cores.

To run some detectors again, give their names with ``--detectors``. The script replaces the results of these detectors and keeps the other results:

.. code-block:: console

   $ python benchmarks/benchmark.py --detectors LODA,RSHash

Commit your code changes before you run the benchmark. If you do not, the commit in the results ends with ``-dirty``.

The ``benchmark`` workflow runs on each pull request that changes ``pysad/``. It runs again only the detectors that the change can affect. Each seed runs in its own job. If an AUROC moves by more than 0.005, the check fails. The output of the check gives the command that updates ``docs/benchmark/``. The check does not compare AUPR, because other platforms round differently. On a small dataset, such a difference can swap one anomaly with one normal point, and that moves AUPR much more than AUROC.
