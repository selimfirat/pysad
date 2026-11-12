Stream your own data
====================

A PySAD detector takes one point at a time. A point is one observation of the stream, as a NumPy array of shape ``(num_features,)``. Any loop that makes such arrays can give data to a detector. This guide shows the common sources of data.

Read a CSV file
---------------

:class:`~pysad.utils.CSVStreamer` reads the file row by row, so that the file can be larger than memory. Select numeric features by zero-based column index. For a file with columns ``timestamp,cpu,memory,latency,label``:

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

:class:`~pysad.utils.PandasStreamer` gives the rows of a DataFrame as NumPy arrays. With ``shuffle=False``, it converts one row at a time. This avoids a NumPy copy of the full frame. It does not give the index. If you need the timestamps, read them from the index of the DataFrame in the same loop:

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

:class:`~pysad.utils.PolarsStreamer` gives the rows of a Polars DataFrame as NumPy arrays. With ``shuffle=False``, it converts fixed-size row buffers. This avoids a NumPy copy of the full frame and keeps NumPy dtypes and timestamp precision. A Polars DataFrame has no index, so drop the timestamp column from the features and read it in the same loop:

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

Convert each message into a NumPy array of floats. This example uses a Kafka consumer. The pattern is the same for every source:

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

To replay the rows in a random order, use ``ArrayStreamer(shuffle=True, random_state=0)``. Shuffle only data that has no time order, because the order of a real stream carries information.

Stream one value per point
--------------------------

Some detectors, such as :class:`~pysad.models.MedianAbsoluteDeviation`, are univariate: they take a point with one feature. Give each value as an array of shape ``(1,)``, not as a plain number:

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

Load a benchmark dataset
------------------------

:class:`~pysad.utils.Data` loads |num_datasets| labeled datasets from the `ODDS <http://odds.cs.stonybrook.edu/>`_ benchmark. To load a dataset, give its file name with the ``.mat`` suffix. The file names are ``arrhythmia.mat``, ``cardio.mat``, ``glass.mat``, ``ionosphere.mat``, ``letter.mat``, ``lympho.mat``, ``mnist.mat``, ``musk.mat``, ``optdigits.mat``, ``pendigits.mat``, ``pima.mat``, ``satellite.mat``, ``satimage-2.mat``, ``shuttle.mat``, ``vertebral.mat``, ``vowels.mat`` and ``wbc.mat``.

.. code-block:: python

   from pysad.utils import Data

   data = Data()
   X, y = data.get_data("arrhythmia.mat")  # arrays with all the points

   for x, label in data.get_iterator("arrhythmia.mat"):  # one point at a time
       ...

``get_iterator`` gives the points in a random order. To get the same order each time, give a seed: ``get_iterator("arrhythmia.mat", random_state=0)``. To keep the order of the file, use ``get_iterator("arrhythmia.mat", shuffle=False)``. ``Data.datasets()`` lists the datasets that ``Data`` can download.

``Data`` downloads a dataset the first time you use it. It caches the file in ``~/.cache/pysad``. To use a different folder, set the ``PYSAD_DATA_DIR`` environment variable. You can also give the folder to ``Data``: ``Data("path/to/folder")``.

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
