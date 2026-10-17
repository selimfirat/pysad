Run a PyOD model on a stream
============================

`PyOD <https://github.com/yzhao062/pyod>`_ has many batch detectors. A batch detector trains on a fixed dataset and cannot learn from one point at a time. PySAD has two wrappers that run a PyOD detector on a stream.

Refit on a sliding window
-------------------------

:class:`~pysad.models.integrations.ReferenceWindowModel` keeps a window of recent points. It scores each point with a detector that it trained on the window, and it trains a new detector on the window at a fixed interval:

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
   The number of points between two trainings. A smaller value follows changes in the data faster, but each training takes time.

``initial_window_X``
   Points to train the first detector on. Without them, the wrapper trains a new detector for each of the first ``window_size`` points, which is slow.

The wrapper passes other keyword arguments, such as ``random_state`` above, to the PyOD detector.

The time per point is not constant: a point that starts a training takes much longer than the other points. If you need a fixed time per point, use a streaming detector from :mod:`pysad.models`.

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

This wrapper is fast, but it does not follow changes in the data. Use it when normal behavior does not change, or as a baseline.
