Run a detector in production
============================

Initialize a detector from history, save its state, handle multiple sources and measure processing time.

The examples on this page use synthetic data. Replace it with your own history and stream:

.. code-block:: python

   import numpy as np

   rng = np.random.default_rng(0)
   X_history = rng.normal(size=(1000, 4))  # points from the past
   stream = rng.normal(size=(500, 4))  # live points, one row at a time
   X_sample = X_history[:200]  # points to measure the time per point

Start from historical data
--------------------------

Fit the detector and calibrator on historical data before processing live points. This initializes the model and the calibrator’s 0-to-1 score scale:

.. code-block:: python

   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   model = LODA(random_state=0)
   calibrator = ConformalProbabilityCalibrator(window_size=1000)

   history_scores = model.fit_score(X_history)  # X_history has shape (num_points, num_features)
   calibrator.fit(history_scores)

``fit_score`` processes rows in order. Representative history can reduce the live warm-up period; check that it reflects current conditions.

Some detectors take initial data in the constructor instead, for example ``initial_window_X`` for :class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.integrations.ReferenceWindowModel`, or ``initial_X`` for :class:`~pysad.models.LocalOutlierProbability`.

Save and restore the state
--------------------------

Save detectors and calibrators together with ``pickle`` to resume from the same state. Pickling is tested for every detector:

.. code-block:: python

   import pickle

   with open("detector.pkl", "wb") as file:
       pickle.dump({"model": model, "calibrator": calibrator}, file)

   with open("detector.pkl", "rb") as file:
       state = pickle.load(file)
   model, calibrator = state["model"], state["calibrator"]

Save the state at a fixed interval, for example every 10,000 points. Also save it when the process stops.

.. warning::

   Load a pickle only with the PySAD version that saved it. The pickle format can change between releases. Pin the version, for example ``pysad==0.6.0``, in your requirements.

   Load pickles only from sources that you trust. A pickle can run code when you load it.

Run one detector per source
---------------------------

If sources have different normal behavior, such as separate servers, keep one detector per source:

.. code-block:: python

   detectors = {}

   def score(source, x):
       if source not in detectors:
           detectors[source] = LODA(random_state=0)
       return detectors[source].fit_score_partial(x)

Each detector needs its own warm-up. To save the state of all sources, pickle the ``detectors`` dictionary.

Keep the input clean
--------------------

- Give every point the same features in the same order. A detector fixes the number of features at the first point.
- Replace missing values before you score a point. Most detectors expect finite values. :class:`~pysad.models.LODA` does not learn from a point with ``NaN`` or infinite values.
- For :class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.RSHash`, set ``feature_mins`` and ``feature_maxes`` from a long period of history. These detectors split the feature space inside this range. Choose a range that holds the values that you expect.

Choose when the detector learns
-------------------------------

``fit_score_partial`` learns from every point. To control learning:

- To freeze the detector after a training period, call only ``score_partial``. The detector then does not follow changes in the data.
- To keep anomalies out of the model, learn only from points that do not raise an alert. :doc:`alerts` shows this loop.

After a permanent change, such as a software release, initialize a new detector using data from after the change.

Control the time per point
--------------------------

Measure the time per point on your own data before you go live:

.. code-block:: python

   import time

   start = time.perf_counter()
   for x in X_sample:
       model.fit_score_partial(x)
   print(f"{(time.perf_counter() - start) / len(X_sample) * 1e6:.0f} µs per point")

Processing time is roughly constant for most detectors. :class:`~pysad.integrations.ReferenceWindowModel` and :class:`~pysad.models.IForestASD` periodically retrain a batch model, causing slower updates.

If a detector is too slow, use fewer ensemble members: ``num_trees`` for the tree-based detectors, ``num_random_cuts`` for LODA, ``n_chains`` for xStream or ``num_components`` for RSHash. You can also project the points to fewer features, as :doc:`pipelines` shows.

Monitor the scores
------------------

Score distribution changes may indicate data problems, such as a broken sensor. Track the recent mean:

.. code-block:: python

   from pysad.statistics import AverageMeter, RunningStatistic

   recent_mean = RunningStatistic(AverageMeter, window_size=1000)

   for x in stream:
       score = model.fit_score_partial(x)
       recent_mean.update(score)

Compare ``recent_mean.get()`` with its value on historical data. If the difference is large, find its cause.

Next steps
----------

- :doc:`alerts` explains thresholds and calibration.
- :doc:`evaluate` shows how to measure a detector before you deploy it.
