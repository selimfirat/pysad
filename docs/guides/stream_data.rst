Stream your own data
====================

Pass each observation as a NumPy array of shape ``(num_features,)``. Any loop yielding these arrays can feed a PySAD detector.

Read a CSV file
---------------

:class:`~pysad.utils.CSVStreamer` reads rows without loading the full file. Select numeric features by zero-based column index. For columns ``timestamp,cpu,memory,latency,label``:

.. code-block:: python

   from pysad.models import LODA
   from pysad.utils import CSVStreamer

   model = LODA()
   streamer = CSVStreamer(feature_columns=[1, 2, 3], label_column=4)
   for x, label in streamer.iter("metrics.csv"):
       score = model.fit_score_partial(x)

The first record is treated as a header by default; use ``has_header=False`` for a headerless file. Omit ``label_column`` to yield feature arrays alone. Omit ``feature_columns`` to use every column except the label. Selected features and labels must be numeric; other columns can contain text. Set ``delimiter`` and ``encoding`` for other CSV formats. Close the iterator if you stop before exhausting it.

Stream a pandas DataFrame
-------------------------

:class:`~pysad.utils.PandasStreamer` yields NumPy rows without the index. With ``shuffle=False``, it converts one row at a time rather than copying the full frame. Read timestamps from the index alongside the stream:

.. code-block:: python

   import pandas as pd

   from pysad.models import LODA
   from pysad.utils import PandasStreamer

   df = pd.read_csv("metrics.csv", index_col="timestamp", parse_dates=True)

   model = LODA()
   for timestamp, x in zip(df.index, PandasStreamer().iter(df)):
       score = model.fit_score_partial(x)

Stream a Polars DataFrame
-------------------------

:class:`~pysad.utils.PolarsStreamer` yields NumPy rows. With ``shuffle=False``, fixed-size buffers avoid copying the full frame and preserve dtypes and timestamp precision. Polars has no index; read timestamps separately from the features:

.. code-block:: python

   import polars as pl

   from pysad.models import LODA
   from pysad.utils import PolarsStreamer

   df = pl.read_csv("metrics.csv", try_parse_dates=True)
   features = df.drop("timestamp")

   model = LODA()
   for timestamp, x in zip(df["timestamp"], PolarsStreamer().iter(features)):
       score = model.fit_score_partial(x)

Read from a message queue or a socket
-------------------------------------

Convert each message to a NumPy float array. For example, with a Kafka consumer:

.. code-block:: python

   import json

   import numpy as np

   from pysad.models import LODA

   model = LODA()
   for message in consumer:  # consumer is your Kafka consumer
       x = np.asarray(json.loads(message.value)["features"], dtype=float)
       score = model.fit_score_partial(x)

Every point must have the same features in the same order.

Stream a NumPy array
--------------------

:class:`~pysad.utils.ArrayStreamer` gives the rows of an array one at a time. If you also give the labels, it gives ``(x, label)`` pairs:

.. code-block:: python

   import numpy as np

   from pysad.models import LODA
   from pysad.utils import ArrayStreamer

   X = np.random.default_rng(0).normal(size=(1000, 3))  # your points
   y = np.zeros(1000, dtype=int)  # your labels: 1 for an anomaly, 0 for normal

   model = LODA()
   for x, label in ArrayStreamer().iter(X, y):
       score = model.fit_score_partial(x)

Use ``ArrayStreamer(shuffle=True, random_state=0)`` for reproducible shuffling. Preserve order when it carries temporal information.

.. _one-dimensional-stream:

Stream one value per point
--------------------------

Univariate detectors, such as :class:`~pysad.models.MedianAbsoluteDeviation`, take each value as an array of shape ``(1,)``:

.. code-block:: python

   import numpy as np

   from pysad.models import MedianAbsoluteDeviation

   values = np.array([10.1, 9.8, 10.3, 10.0, 25.0, 9.9])

   model = MedianAbsoluteDeviation()
   for value in values:
       score = model.fit_score_partial(np.array([value]))
       print(f"{value:5.1f} -> {score:.2f}")

.. code-block:: text

    10.1 -> 0.00
     9.8 -> 1.35
    10.3 -> 0.90
    10.0 -> 0.34
    25.0 -> 67.00
     9.9 -> 0.67

To stream a 1-D array with :class:`~pysad.utils.ArrayStreamer`, first reshape the array to one column: ``values.reshape(-1, 1)``.

Shape and scoring
~~~~~~~~~~~~~~~~~

Pass a feature vector, not a scalar or a row matrix of shape ``(1, 1)``.
Batch input has shape ``(num_instances, num_features)``; ``score(X)`` returns a ``np.float64`` array of shape ``(num_instances,)``.
Keep feature count and order fixed.

``score_partial`` returns a Python ``float`` without learning, even on repeated calls.
Higher scores mean more anomalous points, including when scores are negative.
Use ``fit_score_partial`` to score before learning, or call ``score_partial`` followed by ``fit_partial`` to control updates.
For a fixed model, fit history and then call only ``score_partial``.
See :doc:`/concepts` for scoring order and :doc:`alerts` for selective updates.

Set feature bounds
~~~~~~~~~~~~~~~~~~

:class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.RSHash` require ``feature_mins`` and ``feature_maxes``, each with shape ``(1,)`` for a single feature.
Use distinct limits covering expected feature values, not score ranges or alert thresholds.
For history of shape ``(num_instances, 1)``, compute bounds with ``history.min(axis=0)`` and ``history.max(axis=0)``.
Use data preceding the stream, not future evaluation points.

HalfSpaceTrees uses bounds to build its trees without scaling inputs.
RSHash normalizes internally with ``(x - feature_mins) / (feature_maxes - feature_mins)``.
Neither clips inputs to the bounds.
If you scale inputs, use bounds in the same units.
If the expected range changes, create a detector with new bounds and fit recent history.

Score and calibrate a sensor stream
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example uses known bounds of 0 and 40, generates sensor values with one spike, and scores, calibrates and fits each point.
The calibrator takes a scalar raw score, not the feature vector.
Its ``fit_transform_partial`` includes the current score in the window before computing its rank.
Calibrated values lie between 0 and 1; they are not anomaly probabilities.
With 64 distinct scores, the largest value is ``63 / 64``.
Choose the window and threshold together using :doc:`alerts`.

The example skips alerts for 128 points, covering one model window and one calibration window.
This warm-up is an example choice; the model learns from all points, including alerts.

.. literalinclude:: ../../examples/example_one_dimensional_integration.py
   :language: python

Replace the generated values with your stream.
For benchmark data from :class:`~pysad.utils.Data`, select one feature with ``X[:, :1]`` to preserve the batch shape for ``ArrayStreamer``.

Load a benchmark dataset
------------------------

:class:`~pysad.utils.Data` loads |num_datasets| labeled datasets from the `ODDS <http://odds.cs.stonybrook.edu/>`_ benchmark. To load a dataset, give its file name with the ``.mat`` suffix. The file names are ``arrhythmia.mat``, ``cardio.mat``, ``glass.mat``, ``ionosphere.mat``, ``letter.mat``, ``lympho.mat``, ``mnist.mat``, ``musk.mat``, ``optdigits.mat``, ``pendigits.mat``, ``pima.mat``, ``satellite.mat``, ``satimage-2.mat``, ``shuttle.mat``, ``vertebral.mat``, ``vowels.mat`` and ``wbc.mat``.

.. code-block:: python

   from pysad.utils import Data

   data = Data()
   X, y = data.get_data("arrhythmia.mat")  # arrays with all the points

   for x, label in data.get_iterator("arrhythmia.mat"):  # one point at a time
       ...

``get_iterator`` shuffles by default. Set ``random_state=0`` for reproducible order or ``shuffle=False`` to preserve file order, for example ``get_iterator("arrhythmia.mat", shuffle=False)``. ``Data.datasets()`` lists available datasets.

``Data`` downloads each dataset once to ``~/.cache/pysad``. Override the folder with ``PYSAD_DATA_DIR`` or ``Data("path/to/folder")``.

Credit-card fraud transactions
------------------------------

``Data.datasets("creditcard")`` lists ``creditcard/creditcard.arff``, downloaded from `OpenML 1597, version 1 <https://www.openml.org/d/1597>`_. It contains 284,807 European card transactions over two days in September 2013, including 492 frauds. The source is the research collaboration between Worldline and the Machine Learning Group of ULB; the credited authors are Andrea Dal Pozzolo, Olivier Caelen and Gianluca Bontempi.

``get_data`` preserves the original transaction time order, including the order of transactions with equal times. It returns 30 numeric features in source column order: ``Time`` (seconds since the first transaction), the PCA features ``V1`` through ``V28``, and ``Amount``. ``Class`` is returned separately as ``y``: 1 for fraud and 0 otherwise. No scaling, sampling or shuffling is applied. Use ``ArrayStreamer()`` or ``get_iterator(..., shuffle=False)`` to keep that order; ``get_iterator`` and ``get_stream`` shuffle by default.

The `OpenML metadata <https://www.openml.org/api/v1/json/data/1597>`_ lists the license as "Public". The `original publisher on Kaggle <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>`_ specifies "Database: Open Database, Contents: Database Contents", also exposed by its `metadata API <https://www.kaggle.com/api/v1/datasets/view/mlg-ulb/creditcardfraud>`_. These are the `Open Database License (ODbL) 1.0 <https://opendatacommons.org/licenses/odbl/1-0/>`_ for the database and the `Database Contents License (DbCL) 1.0 <https://opendatacommons.org/licenses/dbcl/1-0/>`_ for its contents. Retain attribution and these license notices when redistributing the data; ODbL also sets share-alike conditions for derived databases. The dataset has its own terms, separate from PySAD's software license.

The first load downloads the ARFF file into the shared dataset cache and verifies its SHA-256 checksum. Later loads reuse that file, including with ``download=False``. PySAD does not bundle the transactions.

.. literalinclude:: ../../examples/example_creditcard.py
   :language: python

Next steps
----------

- :doc:`choose_detector` shows how to find a detector for your data.
- :doc:`alerts` shows how to turn the scores into alerts.
