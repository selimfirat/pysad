Multivariate detectors
======================

This page compares the 10 multivariate detectors of PySAD on |num_datasets| labeled datasets from the `ODDS <http://odds.cs.stonybrook.edu/>`_ repository. A multivariate detector takes a vector of features for each point. Each detector uses its default parameters and scores each point before learning from it (``fit_score_partial``).

Use the results to find two or three candidates, and then compare those candidates on your own data as :doc:`guides/choose_detector` shows. For the univariate detectors, see :doc:`benchmarks/univariate`. For the alerts of the calibrators on the same runs, see :doc:`benchmarks/calibration`. For the evaluation protocol, dataset statistics and scope limitations, see :doc:`benchmarks/methodology`. To reproduce or update these tables, see :doc:`benchmarks/run`.

Metrics
-------

The benchmark measures how well the scores of a detector separate the anomalies from the normal points after the warm-up period. It uses two metrics:

AUROC
   The probability that a random anomaly gets a higher score than a random normal point. A random detector gets 0.5. A perfect detector gets 1.

AUPR
   The average precision over all the score thresholds. A random detector gets the fraction of anomalies in the dataset. A perfect detector gets 1.

:doc:`guides/evaluate` explains the two metrics in more detail.

Summary
-------

The summary has one row for each detector, from the best mean rank to the worst. Detectors with the same mean rank are in the order of their mean AUROC, then their mean AUPR, then their time per point. It has these columns:

Mean rank
   The rank of a detector on a dataset is its position by AUROC, where 1 is the highest AUROC. The mean rank is the average of these ranks over the |num_datasets| datasets.

Mean AUROC and Mean AUPR
   The averages over the |num_datasets| datasets and the 5 seeds.

Time per point
   The median time of one ``fit_score_partial`` call, in milliseconds. The runs shared one machine, so compare these times only with each other.

Commit
   The commit of PySAD that gave the results. Click it to see the code on GitHub.

.. csv-table::
   :file: benchmark/summary.csv
   :header-rows: 1
   :class: benchmark-table

AUROC on each dataset
---------------------

Each value is the mean over the 5 seeds. The highest value on each dataset is in bold. The columns are in the order of the summary. The table uses three short names: HST is :class:`~pysad.models.HalfSpaceTrees`, LoOP is :class:`~pysad.models.LocalOutlierProbability`, and RRCF is :class:`~pysad.models.RobustRandomCutForest`.

.. csv-table::
   :file: benchmark/auroc.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

AUPR on each dataset
--------------------

This table has the same layout as the AUROC table. The AUPR of a random detector is different on each dataset, so compare the values only in the same row.

.. csv-table::
   :file: benchmark/aupr.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

Streams with drift
------------------

The points of the datasets above are in a random order, so they do not change over time. In a real stream, the normal points drift, and an old pattern can come back. Many detectors forget old points to follow such changes, for example with a sliding window. These tables compare the detectors on 8 real streams in their time order:

- KDD Cup 99 http: network connections, with attacks in 10 bursts.
- INSECTS :cite:`souza2020challenges`: 4 streams of optical sensor values of flying insects. A change of the temperature changes the flight of the insects, which gives abrupt, incremental, gradual and reoccurring drift. The southern house mosquito is the anomaly class, as in the ARCUS paper :cite:`yoon2022arcus`.
- SMD :cite:`su2019robust`: 3 server machines, with one point of 38 metrics each minute.

Each stream keeps every k-th point, so that it has at most 10,000 points and keeps all its drifts. The detectors run once, with seed 0. The columns are the same as in the tables above. In the tables of each stream, A, I, IG and IR are the abrupt, incremental, incremental and gradual, and incremental and reoccurring INSECTS streams. :doc:`benchmarks/methodology` describes the streams.

.. csv-table::
   :file: benchmark/drift_summary.csv
   :header-rows: 1
   :class: benchmark-table

AUROC on each stream:

.. csv-table::
   :file: benchmark/drift_auroc.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

AUPR on each stream:

.. csv-table::
   :file: benchmark/drift_aupr.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table
