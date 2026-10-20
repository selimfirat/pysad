Multivariate detectors
======================

This page compares the 10 multivariate detectors on |num_datasets| datasets from the `ODDS <http://odds.cs.stonybrook.edu/>`_ repository. The points of each dataset are in a random order, so they do not change over time. :doc:`drift` compares the same detectors on streams that change over time. :doc:`methodology` explains the steps of a run and the metrics.

Summary
-------

The summary has one row for each detector, from the best mean rank to the worst. The best value of each column is in bold. The summary has these columns:

Mean rank
   The rank of a detector on a dataset is its position by AUROC, where 1 is the highest AUROC. The mean rank is the average of these ranks over the |num_datasets| datasets. Detectors with the same mean rank are in the order of their mean AUROC, then their mean AUPR, then their time per point.

Mean AUROC and Mean AUPR
   The averages over the |num_datasets| datasets.

Time per point
   The median time of one ``fit_score_partial`` call, in milliseconds. All the runs used one computer, so compare these times only with each other.

Commit
   The commit of PySAD that gave the results.

.. csv-table::
   :file: ../benchmark/summary.csv
   :header-rows: 1
   :class: benchmark-table best-columns

AUROC on each dataset
---------------------

The highest value on each dataset is in bold. The columns are in the order of the summary. HST is :class:`~pysad.models.HalfSpaceTrees`, LoOP is :class:`~pysad.models.LocalOutlierProbability`, and RRCF is :class:`~pysad.models.RobustRandomCutForest`.

.. csv-table::
   :file: ../benchmark/auroc.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

AUPR on each dataset
--------------------

The AUPR of a random detector is the fraction of anomalies in the dataset, which is different on each dataset. Thus, compare the values only in the same row.

.. csv-table::
   :file: ../benchmark/aupr.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table
