.. meta::
   :description: Choose between the NAB, Yahoo S5 and ODDS streaming dataset loaders of pysad.utils.Data by the kind of data, the labels, the size and the license.

Choose a streaming dataset
==========================

PySAD loads labeled streams from three public collections: NAB, Yahoo S5 and ODDS. This guide helps you choose the collection that fits your data.

NAB and Yahoo S5 are univariate: one value per point. ODDS is multivariate: several features per point. Choose by the shape of your data, the label style you need, and the license you can comply with.

NAB, Yahoo S5 and ODDS at a glance
----------------------------------

.. list-table:: Choose a collection by the kind of data, the labels, the size and the license
   :header-rows: 1
   :widths: 20 26 26 28

   * -
      - NAB
      - Yahoo S5
      - ODDS
   * - Kind of data
      - One value per point (univariate)
      - One value per point (univariate)
      - Several features per point (multivariate)
   * - Series
      - 58 series
      - 67 series
      - 17 datasets
   * - Labels
      - Anomaly windows
      - One label per point
      - One label column per file
   * - Size
      - Small to medium series
      - Medium series, one point per hour
      - Small to large tabular datasets
   * - License
      - MIT
      - Yahoo Data Sharing Agreement
      - Varies by dataset
   * - Best for
      - Server metrics, CPU, traffic, tweets
      - Production traffic of a Kubernetes cluster
      - Tabular anomaly benchmarks

Load each collection
--------------------

NAB

.. code-block:: python

   from pysad.utils import Data

   data = Data()
   X, y = data.get_data("nab/realKnownCause/nyc_taxi.csv")

Yahoo S5

.. code-block:: python

   from pysad.utils import Data

   data = Data()
   X, y = data.get_data("yahoo/real_1.csv")

ODDS

.. code-block:: python

   from pysad.utils import Data

   data = Data()
   X, y = data.get_data("cardio.mat")

Stream each point with :meth:`~pysad.utils.Data.get_iterator`, which shuffles by default. Set ``shuffle=False`` to preserve file order, which matters when the order carries time. See :doc:`stream_data` for the full how-to of each loader and for :meth:`~pysad.utils.Data.get_stream`, which gives the stream of the PySAD benchmark.

NAB gives each point as ``[timestamp, value]`` with ``timestamps=True``. Yahoo S5 gives one row per hour with ``timestamp,value,is_anomaly``. ODDS gives a ``.mat`` file with features and labels already separated.

Labels
------

Label 1 always marks the anomaly and label 0 marks the normal point, except where a loader notes otherwise.

- NAB labels the points inside an anomaly window from ``combined_windows.json``.
- Yahoo S5 labels each point with its own ``is_anomaly`` column.
- ODDS keeps one label column per file. For ``vertebral.mat``, PySAD flips the labels so the unhealthy patients are the anomaly class.

See :doc:`dataset_inventory` for the label format, source, license and checksum of every loader.

License
-------

- NAB is under the MIT license at the commit PySAD uses. Earlier commits of NAB are under AGPL-3.0.
- Yahoo S5 requires a signed Yahoo Data Sharing Agreement for academic, non-commercial research, and Yahoo prohibits redistribution.
- ODDS licenses vary by dataset. The original source is the ODDS library of Stony Brook University, and PySAD downloads the PyOD mirror at a fixed commit.

Check :meth:`~pysad.utils.Data.metadata` for the source URL, license and SHA-256 checksum of each file before you download it.

How to choose
-------------

- You have one series of values and want anomaly windows: start with NAB.
- You have one series of values with one point per hour and per-point labels: start with Yahoo S5.
- You have several features per point: start with ODDS, for example ``cardio.mat``.
- You cannot sign a data sharing agreement: avoid Yahoo S5 and use NAB or ODDS.
- You want a multivariate tabular benchmark: use ODDS.

Next steps
----------

- :doc:`stream_data` shows how to load and stream each collection.
- :doc:`choose_detector` shows how to find a detector for your data.
- :doc:`evaluate` shows how to score and compare detectors on labeled data.
