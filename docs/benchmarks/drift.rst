Streams with drift
==================

In a real stream, the normal points change over time, and an old pattern can come back. This change is called drift. Many detectors forget old points to follow drift, for example with a sliding window. This page compares the 10 multivariate detectors on 8 real streams in their time order:

- KDD Cup 99 http: network connections, with attacks in 10 bursts.
- INSECTS: 4 streams of optical sensor values of flying insects. A change of the temperature changes how the insects fly. The streams have abrupt (A), incremental (I), incremental and gradual (IG), and incremental and reoccurring (IR) drift.
- SMD: 3 server machines, with one point of 38 metrics each minute.

The tables have the same columns as the tables on :doc:`multivariate`. The best value of each column is in bold. :doc:`datasets` describes the streams, and :doc:`methodology` explains the steps of a run.

Summary
-------

.. csv-table::
   :file: ../benchmark/drift_summary.csv
   :header-rows: 1
   :class: benchmark-table best-columns

AUROC on each stream
--------------------

.. csv-table::
   :file: ../benchmark/drift_auroc.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

AUPR on each stream
-------------------

.. csv-table::
   :file: ../benchmark/drift_aupr.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table
