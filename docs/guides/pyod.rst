Run a PyOD model on a stream
============================

Run `PyOD <https://github.com/yzhao062/pyod>`_ batch detectors on a stream with either wrapper below. Install PyOD separately with ``pip install pyod``.

The examples on this page use a synthetic stream. Replace ``X`` with your own data:

.. code-block:: python

   import numpy as np

   from pysad.utils import ArrayStreamer

   X = np.random.default_rng(42).normal(size=(2000, 4))
   streamer = ArrayStreamer()

Refit on a sliding window
-------------------------

:class:`~pysad.integrations.ReferenceWindowModel` trains on a recent window and scores incoming points. It retrains every ``sliding_size`` points:

.. code-block:: python

   from pyod.models.iforest import IForest

   from pysad.integrations import ReferenceWindowModel

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
   Points between training runs. Smaller values adapt faster but retrain more often.

``initial_window_X``
   Initial training points. If omitted, the wrapper trains each time the number of stored points doubles, so the first window costs about log2(window_size) fits.

The wrapper passes other keyword arguments, such as ``random_state`` above, to the PyOD detector.

Retraining makes some updates much slower. For steadier processing time, use a streaming detector from :mod:`pysad.models`.

:class:`~pysad.models.IForestASD` is a ``ReferenceWindowModel`` with an Isolation Forest, as in its paper.

Train once
----------

:class:`~pysad.integrations.OneFitModel` trains the detector once, on initial data, and then only scores:

.. code-block:: python

   from pyod.models.knn import KNN

   from pysad.integrations import OneFitModel

   model = OneFitModel(KNN, initial_X=X[:500], n_neighbors=10)

   for x in streamer.iter(X[500:]):
       score = model.fit_score_partial(x)

Use this fixed model when normal behavior stays stable, or as a baseline for comparing adaptive detectors.

Next steps
----------

- :doc:`choose_detector` helps you choose between the wrapper and a streaming detector.
- :doc:`production` explains the time per point in production.
