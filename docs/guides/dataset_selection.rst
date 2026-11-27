Choose a streaming dataset
==========================

:class:`~pysad.utils.Data` downloads labeled datasets, verifies every download against a SHA-256 checksum, and caches the files in ``~/.cache/pysad``. This guide compares the three collections that ship labeled streaming series: NAB, Yahoo S5, and ODDS. Pick the collection that matches your data.

.. list-table:: NAB, Yahoo S5 and ODDS at a glance
   :header-rows: 1
   :widths: 16 26 28 30

   * -
     - NAB
     - Yahoo S5
     - ODDS
   * - Series
     - 58 univariate series. 47 are real-world and 11 are artificial.
     - 67 univariate real production-traffic series.
     - 17 multivariate datasets.
   * - Features
     - One feature: the value.
     - One feature: the value.
     - Two or more features per dataset.
   * - Labels
     - Label 1 marks the points in an anomaly window.
     - Label 1 marks an anomaly point. The last column holds the label.
     - Label 1 marks the anomaly class. For vertebral.mat the labels are flipped.
   * - Order
     - File order. get_iterator shuffles by default.
     - File order. Rows keep file order. One row per hour.
     - File order for the arrays in each .mat file.
   * - Timestamps
     - Yes, with timestamps=True. NAB gives local times read as UTC.
     - Yes, the first column counts hours. It is not a feature.
     - No timestamps.
   * - License
     - MIT at commit ea702d75. Earlier commits of NAB are under AGPL-3.0.
     - Yahoo Webscope Data Sharing Agreement. Academic, non-commercial research only. Yahoo prohibits redistribution.
     - Varies by dataset. PySAD downloads the PyOD mirror at a fixed commit.

NAB — univariate metrics with anomaly windows
---------------------------------------------

NAB (Numenta Anomaly Benchmark) holds 58 univariate series, for example ``nab/realKnownCause/nyc_taxi.csv``. Each series is a CSV with the header ``timestamp,value``. The points in an anomaly window have the label 1, from ``combined_windows.json``. Use NAB for univariate server and cloud metrics with labeled anomaly windows, for example CPU utilization, request latency and network traffic.

.. code-block:: python

   from pysad.utils import Data

   data = Data()
   X, y = data.get_data("nab/realKnownCause/nyc_taxi.csv")

With ``timestamps=True``, :meth:`~pysad.utils.Data.get_data` returns each point as ``[timestamp, value]``, as :class:`~pysad.models.HTM` takes it. NAB gives local times without a time zone, and the timestamps read them as UTC.

:meth:`~pysad.utils.Data.datasets` with ``"nab"`` lists all 58 series. With ``"nab-real-world"`` it lists the 47 real-world series only, excluding the artificial series under ``artificialNoAnomaly`` and ``artificialWithAnomaly``.

Yahoo S5 — hourly production traffic
------------------------------------

Yahoo S5 A1Benchmark holds 67 univariate real production-traffic series, for example ``yahoo/real_1.csv``. Each row contains ``timestamp,value,is_anomaly``. The value column is the only feature, label 1 marks an anomaly, and rows keep file order. One row counts one hour.

.. code-block:: python

   from pysad.utils import Data

   data = Data()
   X, y = data.get_data("yahoo/real_1.csv")

Yahoo S5 requires a signed Yahoo Webscope Data Sharing Agreement for academic, non-commercial research. Yahoo prohibits redistribution. The loader downloads from a public mirror at a fixed commit. The mirror does not grant a separate data license. Get the agreement before you use this loader.

:meth:`~pysad.utils.Data.datasets` with ``"yahoo"`` lists all 67 series. The loader excludes the synthetic A2, A3, and A4 subsets.

ODDS — multivariate outlier detection
-------------------------------------

ODDS holds 17 multivariate datasets, for example ``cardio.mat``. Each dataset has two or more features, and label 1 marks the anomaly class. For ``vertebral.mat``, PySAD flips the labels from the ODDS file: ODDS marks the 30 healthy patients as anomalies, so the returned labels make the 210 unhealthy patients the anomalies.

.. code-block:: python

   from pysad.utils import Data

   data = Data()
   X, y = data.get_data("cardio.mat")

Use ODDS for multivariate outlier detection, where each point has several features and you want to score the whole point. :meth:`~pysad.utils.Data.datasets` with ``"odds"`` lists the 17 file names.

Which loader to choose
----------------------

- Your stream has one value per point and labeled anomaly windows: start with NAB.
- Your stream has one value per point, one row per hour, and you can sign the Yahoo agreement: use Yahoo S5.
- Your stream has several features per point: use ODDS.
- You want only real measured series and no generated ones: use NAB with ``nab-real-world``.
- You want to benchmark a detector on a public labeled stream: use NAB or Yahoo S5.
- You want multivariate outlier detection on a fixed set of datasets: use ODDS.

Example: score an ODDS stream one point at a time
-------------------------------------------------

.. code-block:: python

   import numpy as np

   from pysad.models import LODA
   from pysad.utils import Data

   data = Data()
   X, y = data.get_data("cardio.mat")

   model = LODA(random_state=0)
   scores = []
   for x, label in data.get_iterator("cardio.mat", shuffle=False, random_state=0):
       scores.append(model.fit_score_partial(x))

   print(f"Scored {len(scores)} points. Anomaly rate: {np.mean(y):.3f}")

:meth:`~pysad.utils.Data.get_iterator` shuffles by default. Pass ``shuffle=False`` to preserve file order, which matters when order carries time.

See also
--------

- :doc:`dataset_inventory`: every loader with its source, license, checksum and label format.
- :doc:`stream_data`: connect CSV files, DataFrames and message queues to a detector.
