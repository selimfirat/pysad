Run a PyOD model on a stream
============================

`PyOD <https://github.com/yzhao062/pyod>`_ has many batch detectors. A batch detector trains on a fixed dataset and cannot learn from one point at a time. PySAD has two wrappers that run a PyOD detector on a stream.

The examples on this page use a synthetic stream. Replace ``X`` with your own data:

.. code-block:: python

   import numpy as np

   from pysad.utils import ArrayStreamer

   X = np.random.default_rng(42).normal(size=(2000, 4))
   streamer = ArrayStreamer()

Refit on a sliding window
-------------------------

:class:`~pysad.models.integrations.ReferenceWindowModel` keeps a window of recent points. It trains a detector on the window and scores each new point with that detector. Every ``sliding_size`` points, it trains a new detector on the current window:

.. code-block:: python

   from pyod.models.iforest import IForest

   from pysad.models.integrations import ReferenceWindowModel

   model = ReferenceWindowModel(
       IForest,
       window_size=500,
       sliding_size=100,
       initial_window_X=X[:500],
       random_state=0,
   )

   for x in streamer.iter(X[500:]):
       score = model.fit_score_partial(x)

``window_size``
   The number of recent points that the detector trains on.

``sliding_size``
   The number of points between two training runs. A smaller value follows changes in the data faster, but each training run takes time.

``initial_window_X``
   Points to train the first detector on. If you do not give these points, the wrapper trains a new detector for each of the first ``window_size`` points, which is slow.

The wrapper passes other keyword arguments, such as ``random_state`` above, to the PyOD detector.

The time per point is not constant. A point that starts a training run takes much longer than the other points. If you need a fixed time per point, use a streaming detector from :mod:`pysad.models`.

:class:`~pysad.models.IForestASD` is a ``ReferenceWindowModel`` with an Isolation Forest, as in its paper.

Train once
----------

:class:`~pysad.models.integrations.OneFitModel` trains the detector once, on initial data, and then only scores:

.. code-block:: python

   from pyod.models.knn import KNN

   from pysad.models.integrations import OneFitModel

   model = OneFitModel(KNN, initial_X=X[:500], n_neighbors=10)

   for x in streamer.iter(X[500:]):
       score = model.fit_score_partial(x)

This wrapper is fast, but it does not follow changes in the data. If normal behavior does not change, use it. You can also use it as a baseline, a reference result to compare other detectors with.

Next steps
----------

- :doc:`choose_detector` helps you choose between the wrapper and a streaming detector.
- :doc:`production` explains the time per point in production.
