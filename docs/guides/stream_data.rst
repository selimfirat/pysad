Stream your own data
====================

A PySAD detector takes one point at a time as a NumPy array of shape ``(num_features,)``. Any loop that makes such arrays works. This guide shows the common sources.

From a CSV file
---------------

Read the file row by row, so that the file can be larger than memory:

.. code-block:: python

   import csv

   import numpy as np

   from pysad.models import LODA

   model = LODA()

   with open("metrics.csv", newline="") as file:
       for row in csv.DictReader(file):
           x = np.array([float(row["cpu"]), float(row["memory"]), float(row["latency"])])
           score = model.fit_score_partial(x)

From a pandas DataFrame
-----------------------

:class:`~pysad.utils.PandasStreamer` gives the rows of a DataFrame as NumPy arrays. If you need the timestamps, keep the index separately:

.. code-block:: python

   import pandas as pd

   from pysad.models import LODA
   from pysad.utils import PandasStreamer

   df = pd.read_csv("metrics.csv", index_col="timestamp", parse_dates=True)

   model = LODA()
   for timestamp, x in zip(df.index, PandasStreamer().iter(df)):
       score = model.fit_score_partial(x)

From a message queue or a socket
--------------------------------

Convert each message into a NumPy array of floats. This example uses a Kafka consumer, but the pattern is the same for every source:

.. code-block:: python

   import json

   import numpy as np

   for message in consumer:
       x = np.asarray(json.loads(message.value)["features"], dtype=float)
       score = model.fit_score_partial(x)

Every point must have the same features in the same order.

From a NumPy array
------------------

:class:`~pysad.utils.ArrayStreamer` gives the rows of an array one at a time. With labels, it gives ``(x, label)`` pairs:

.. code-block:: python

   from pysad.models import LODA
   from pysad.utils import ArrayStreamer

   model = LODA()
   for x, label in ArrayStreamer().iter(X, y):
       score = model.fit_score_partial(x)

To replay the rows in a random order, use ``ArrayStreamer(shuffle=True, random_state=0)``. The order of a real stream carries information, so shuffle only data that has no time order.

One value per point
-------------------

Univariate detectors, such as :class:`~pysad.models.MedianAbsoluteDeviation`, take a point with one feature. Give each value as an array of shape ``(1,)``, not as a plain number:

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

To stream a 1-D array with :class:`~pysad.utils.ArrayStreamer`, reshape it to one column first: ``values.reshape(-1, 1)``.

Benchmark datasets
------------------

:class:`~pysad.utils.Data` loads 17 labeled datasets from the `ODDS <http://odds.cs.stonybrook.edu/>`_ benchmark. Pass the file name, with the ``.mat`` suffix: ``arrhythmia.mat``, ``cardio.mat``, ``glass.mat``, ``ionosphere.mat``, ``letter.mat``, ``lympho.mat``, ``mnist.mat``, ``musk.mat``, ``optdigits.mat``, ``pendigits.mat``, ``pima.mat``, ``satellite.mat``, ``satimage-2.mat``, ``shuttle.mat``, ``vertebral.mat``, ``vowels.mat`` and ``wbc.mat``.

.. code-block:: python

   from pysad.utils import Data

   data = Data()
   X, y = data.get_data("arrhythmia.mat")  # arrays with all the points

   for x, label in data.get_iterator("arrhythmia.mat"):  # one point at a time
       ...

``get_iterator`` gives the points in a random order. For the order of the file, use ``get_iterator("arrhythmia.mat", shuffle=False)``.

``Data`` downloads a dataset the first time you use it. It caches the file in ``~/.cache/pysad``. To use a different folder, set the ``PYSAD_DATA_DIR`` environment variable, or pass the folder: ``Data("path/to/folder")``.

Next steps
----------

- :doc:`choose_detector` finds a detector for your data.
- :doc:`alerts` turns the scores into alerts.
