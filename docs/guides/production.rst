Run a detector in production
============================

This guide shows how to start a detector from historical data and how to save and restore its state. It also shows how to run one detector per data source and how to control the time per point.

The examples on this page use synthetic data. Replace it with your own history and stream:

.. code-block:: python

   import numpy as np

   rng = np.random.default_rng(0)
   X_history = rng.normal(size=(1000, 4))  # points from the past
   stream = rng.normal(size=(500, 4))  # live points, one row at a time
   X_sample = X_history[:200]  # points to measure the time per point

Start from historical data
--------------------------

A new detector has no model of normal behavior, so its first scores are not reliable. A calibrator maps scores to values between 0 and 1. If you have historical data, fit the detector and the calibrator to it before the live stream starts:

.. code-block:: python

   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   model = LODA(random_state=0)
   calibrator = ConformalProbabilityCalibrator(window_size=1000)

   history_scores = model.fit_score(X_history)  # X_history has shape (num_points, num_features)
   calibrator.fit(history_scores)

``fit_score`` processes the rows in order, as a stream. After this step, the first live points get reliable scores. You do not need a warm-up period, a first part of the stream with unreliable scores.

Some detectors take initial data in the constructor instead, for example ``initial_window_X`` for :class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.integrations.ReferenceWindowModel`, or ``initial_X`` for :class:`~pysad.models.LocalOutlierProbability`.

Save and restore the state
--------------------------

You can save PySAD detectors and calibrators with ``pickle``. The PySAD tests make sure that this works for every detector. Save the detector and the calibrator together, so that a restart continues from the same state:

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

Normal behavior is usually different for each data source, for example for each server. If the stream mixes data from several sources, keep one detector for each source:

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

If you call ``fit_score_partial``, the detector learns from every point and follows slow changes in the data. You can also use two other modes:

- To freeze the detector after a training period, call only ``score_partial``. The detector then does not follow changes in the data.
- To keep anomalies out of the model, learn only from points that do not raise an alert. :doc:`alerts` shows this loop.

If the data changes permanently, for example after a new software release, make a new detector. Start it from data after the change, because the old normal behavior no longer applies.

Control the time per point
--------------------------

Measure the time per point on your own data before you go live:

.. code-block:: python

   import time

   start = time.perf_counter()
   for x in X_sample:
       model.fit_score_partial(x)
   print(f"{(time.perf_counter() - start) / len(X_sample) * 1e6:.0f} µs per point")

Most detectors take about the same time for every point. :class:`~pysad.models.MedianAbsoluteDeviation` is an exception: its time per point grows with the stream. :class:`~pysad.models.integrations.ReferenceWindowModel` and :class:`~pysad.models.IForestASD` train a new batch model at a fixed interval, so a few points take much longer than the others.

If a detector is too slow, use fewer ensemble members: ``num_trees`` for the tree-based detectors, ``num_random_cuts`` for LODA, ``n_chains`` for xStream or ``num_components`` for RSHash. You can also project the points to fewer features, as :doc:`pipelines` shows.

Monitor the scores
------------------

A change in the score distribution can show a problem with the data, for example a broken sensor. Track the mean of the recent scores with a running statistic:

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
