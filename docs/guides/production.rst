Run a detector in production
============================

This guide shows how to start a detector from historical data, save and restore its state, run one detector per data source and keep the time per point under control.

The examples on this page use synthetic data. Replace it with your own history and stream:

.. code-block:: python

   import numpy as np

   rng = np.random.default_rng(0)
   X_history = rng.normal(size=(1000, 4))  # points from the past
   stream = rng.normal(size=(500, 4))  # live points, one row at a time
   X_sample = X_history[:200]  # points to measure the time per point

Start from historical data
--------------------------

A new detector knows nothing, and its first scores are not reliable. If you have historical data, fit the detector and the calibrator to it before the live stream starts:

.. code-block:: python

   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   model = LODA(random_state=0)
   calibrator = ConformalProbabilityCalibrator(window_size=1000)

   history_scores = model.fit_score(X_history)  # X_history has shape (num_points, num_features)
   calibrator.fit(history_scores)

``fit_score`` processes the rows in order, as a stream. After this step, the first live points get stable scores, and you do not need a warm-up period.

Some detectors take initial data in the constructor instead, for example ``initial_window_X`` for :class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.integrations.ReferenceWindowModel`, or ``initial_X`` for :class:`~pysad.models.LocalOutlierProbability`.

Save and restore the state
--------------------------

PySAD detectors and calibrators can be pickled, and the test suite checks this for every detector. Save the detector and the calibrator together, so that a restart continues from the same state:

.. code-block:: python

   import pickle

   with open("detector.pkl", "wb") as file:
       pickle.dump({"model": model, "calibrator": calibrator}, file)

   with open("detector.pkl", "rb") as file:
       state = pickle.load(file)
   model, calibrator = state["model"], state["calibrator"]

Save the state at a fixed interval, for example every 10,000 points, and when the process stops.

.. warning::

   Load a pickle only with the PySAD version that saved it. The pickle format can change between releases. Pin the version, for example ``pysad==0.6.0``, in your requirements.

   Load pickles only from sources that you trust. A pickle can run code when it is loaded.

Run one detector per source
---------------------------

If the stream mixes data from several sources, for example several servers, normal behavior is usually different for each source. Keep one detector for each source:

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
- For :class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.RSHash`, set ``feature_mins`` and ``feature_maxes`` from a long period of history. These detectors split the space inside this range, so choose a range that holds the values that you expect.

Learn or only score
-------------------

By default, the detector learns from every point with ``fit_score_partial``, so it follows slow changes in the data. Two other modes are useful:

- To freeze the detector after a training period, call only ``score_partial``. The detector then does not follow changes in the data.
- To keep anomalies out of the model, learn only from points that do not raise an alert. :doc:`alerts` shows this loop.

After a permanent change, for example a new software release, the old normal behavior no longer applies. Make a new detector and start it from data after the change.

Control the time per point
--------------------------

Measure the time per point on your own data before you go live:

.. code-block:: python

   import time

   start = time.perf_counter()
   for x in X_sample:
       model.fit_score_partial(x)
   print(f"{(time.perf_counter() - start) / len(X_sample) * 1e6:.0f} µs per point")

Most detectors take about the same time for every point. :class:`~pysad.models.MedianAbsoluteDeviation` is an exception: its time per point grows with the stream. :class:`~pysad.models.integrations.ReferenceWindowModel` and :class:`~pysad.models.IForestASD` train a new batch model at a fixed interval, so a few points take much longer than the others. To make the detector faster, use fewer ensemble members (``num_trees``, ``num_random_cuts``, ``num_components``) or project the points to fewer features, as :doc:`pipelines` shows.

Monitor the scores
------------------

A change in the score distribution can show a problem with the data, for example a broken sensor. Track the scores with running statistics:

.. code-block:: python

   from pysad.statistics import AverageMeter, RunningStatistic

   recent_mean = RunningStatistic(AverageMeter, window_size=1000)

   for x in stream:
       score = model.fit_score_partial(x)
       recent_mean.update(score)

Compare ``recent_mean.get()`` with its value on historical data, and investigate a large change.

Next steps
----------

- :doc:`alerts` explains thresholds and calibration.
- :doc:`evaluate` shows how to measure a detector before you deploy it.
