Datasets
========

This page describes the data of each part of the benchmark. :class:`~pysad.utils.Data` downloads all of it on first use. :doc:`methodology` explains which points each run keeps.

ODDS
----

The ``odds`` part uses |num_datasets| datasets from the `ODDS <http://odds.cs.stonybrook.edu/>`_ repository (Outlier Detection DataSets). The column "Points used" gives the number of points after the limit of 5,000, and "Points evaluated" the number after the warm-up. :meth:`Data.get_stream <pysad.utils.Data.get_stream>` gives a stream exactly as the benchmark uses it, for example ``Data().get_stream("cardio.mat", random_state=0, max_points=5000, scale=True)``.

.. csv-table::
   :file: ../benchmark/datasets.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

Streams with drift
------------------

The ``drift`` part uses 8 streams from three sources:

- ``Data.datasets("kddcup99")`` gives the http connections of logged in users from the 10% file of KDD Cup 99, as scikit-learn's ``fetch_kddcup99`` does. Its 3 features are the logarithms of the duration and of the bytes in each direction. The attacks are the anomalies.
- ``Data.datasets("insects")`` gives the 4 INSECTS streams :cite:`souza2020challenges` as the ARCUS paper :cite:`yoon2022arcus` made them. The anomalies are the points of the southern house mosquito, 1.2% of each stream.
- ``Data.datasets("smd")`` gives the test halves of the 28 machines of SMD :cite:`su2019robust`. The benchmark uses the first machine of each of the 3 groups of machines. The first half of each machine has no labels, so the benchmark does not use it.

The column "Points used" gives the number of points after the thinning.

.. csv-table::
   :file: ../benchmark/drift_streams.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

NAB and UCR
-----------

NAB has 58 series, and 52 of them have anomalies. Each point in a window of NAB has the label 1. 14 series are seasonal: the correlation of their values with the values one day earlier is at least 0.5. Their period is one day, which is 24, 48 or 288 points, because these series have a point every hour, every half hour or every 5 minutes.

The UCR archive has 250 series with 19.4 million points. The benchmark uses the 84 series with at most 20,000 points, so that a run stays short. The archive has no license that lets PySAD copy it, so :class:`~pysad.utils.Data` downloads it from the site of its authors.
