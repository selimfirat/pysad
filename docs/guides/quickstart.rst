.. meta::
   :description: Quick start guide for streaming anomaly detection with PySAD: install, score points one at a time, measure AUROC, and raise alerts.

Quick start
===========

Score a stream point by point, measure the scores, and raise alerts. This page shows the minimal PySAD loop. Allow five minutes.

Install PySAD
-------------

Install PySAD with pip:

.. code-block:: console

   $ pip install pysad

For conda and other installation options, see :doc:`/installation`.

Make a stream
-------------

Each point is a NumPy array of shape ``(num_features,)``. This stream has 2,000 points with 4 features. Every 50th point is an anomaly.

.. code-block:: python

   import numpy as np

   from pysad.utils import ArrayStreamer

   rng = np.random.default_rng(42)
   X = rng.normal(size=(2000, 4))
   y = np.zeros(2000, dtype=int)
   y[::50] = 1
   X[y == 1, :2] += 3

   streamer = ArrayStreamer()

``X`` holds the points. ``y`` holds labels: 1 for an anomaly and 0 for normal. Only the metrics use the labels.

Score each point
----------------

Create a detector and score each point before learning from it:

.. code-block:: python

   from pysad.models import LODA

   model = LODA(random_state=0)

   scores = []
   for x in streamer.iter(X):
       score = model.fit_score_partial(x)
       scores.append(score)

:meth:`~pysad.core.BaseModel.fit_score_partial` returns a Python ``float``. Higher means more anomalous. The detector learns without labels.

Compare the average score of normal points and anomalies:

.. code-block:: python

   scores = np.array(scores)
   print(f"normal: {scores[y == 0].mean():.2f}")
   print(f"anomalies: {scores[y == 1].mean():.2f}")

Measure the detector
--------------------

Measure the ranking with AUROC. A perfect ranking gets 1.0. Random scores get 0.5 on average.

.. code-block:: python

   from pysad.evaluation import AUROCMetric

   model = LODA(random_state=0)
   metric = AUROCMetric()

   for x, label in streamer.iter(X, y):
       score = model.fit_score_partial(x)
       metric.update(label, score)

   print(f"AUROC: {metric.get():.3f}")

Scoring before learning is prequential evaluation. Each score is computed on a point the model has not seen yet. See :doc:`/concepts`.

Raise alerts
------------

Raw score scales differ between detectors. Conformal calibration ranks each score against recent scores, so one threshold works across detectors:

.. code-block:: python

   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   model = LODA(random_state=0)
   calibrator = ConformalProbabilityCalibrator(window_size=500)

   alerts = []
   for t, x in enumerate(streamer.iter(X)):
       score = model.fit_score_partial(x)
       calibrated = calibrator.fit_transform_partial(score)
       if t >= 200 and calibrated >= 0.99:
           alerts.append(t)

   print(f"{len(alerts)} alerts")

A value of 0.99 means that 99% of recent scores are lower. The calibrator keeps up to 500 scores. We skip alerts for the first 200 points while the model and calibrator warm up. Choose a threshold and window for your data using :doc:`/guides/alerts`.

Next steps
----------

- :doc:`/guides/stream_data` streams CSV files, DataFrames and message queues.
- :doc:`/guides/choose_detector` helps you choose a detector for your data.
- :doc:`/guides/evaluate` measures performance over time.
- :doc:`/concepts` explains the streaming model and prequential evaluation.
- :ref:`api` lists every class and parameter.
