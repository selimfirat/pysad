Getting started
===============

In this tutorial, you build a small anomaly detector for a data stream. An anomaly detector gives each data point a score that shows how unusual the point is. You score each point when it arrives, measure how well the detector works, and raise alerts. The tutorial takes about ten minutes.

You need Python 3.10 or newer. You do not need to know about anomaly detection.

Install PySAD
-------------

Install PySAD from PyPI:

.. code-block:: console

   $ pip install pysad

Make a stream
-------------

A stream is a sequence of data points that arrive one at a time. Each point is a NumPy array of features (measured values). In a real system, the points come from sensors, logs or a message queue. In this tutorial, you generate the stream with code so that you know which points are anomalies.

The stream has 2,000 points with 4 features. Every 50th point is an anomaly: the code adds 3 to its first two features.

.. code-block:: python

   import numpy as np

   rng = np.random.default_rng(42)
   X = rng.normal(size=(2000, 4))
   y = np.zeros(2000, dtype=int)
   y[::50] = 1
   X[y == 1, :2] += 3

``X`` holds the points and ``y`` holds the labels: 1 for an anomaly and 0 for a normal point. The detector never sees ``y``. You use ``y`` only to measure the detector.

Score each point
----------------

Make a detector. Then give it the stream one point at a time. This tutorial uses :class:`~pysad.models.LODA`, a fast detector for data with several features.

.. code-block:: python

   from pysad.models import LODA
   from pysad.utils import ArrayStreamer

   model = LODA(random_state=0)
   streamer = ArrayStreamer()

   scores = []
   for x in streamer.iter(X):
       score = model.fit_score_partial(x)
       scores.append(score)

:class:`~pysad.utils.ArrayStreamer` replays an array as a stream, one row at a time. For each point, :meth:`~pysad.core.BaseModel.fit_score_partial` gives the point an anomaly score, and then the detector learns from the point. The score is a Python ``float``. A higher score means a more anomalous point.

Compare the average score of the normal points and of the anomalies:

.. code-block:: python

   scores = np.array(scores)
   print(f"normal: {scores[y == 0].mean():.2f}")
   print(f"anomalies: {scores[y == 1].mean():.2f}")

.. code-block:: text

   normal: 1.50
   anomalies: 2.63

The anomalies get higher scores. The detector learned what normal data looks like from the stream itself, without labels.

Measure the detector
--------------------

A metric is a number that shows how well the scores separate anomalies from normal points. AUROC (the area under the ROC curve) is the probability that a random anomaly gets a higher score than a random normal point. It is 1.0 for a perfect detector and 0.5 for random scores.

PySAD metrics update one point at a time, so you can measure the detector while the stream runs:

.. code-block:: python

   from pysad.evaluation import AUROCMetric

   model = LODA(random_state=0)
   metric = AUROCMetric()

   for x, label in streamer.iter(X, y):
       score = model.fit_score_partial(x)
       metric.update(label, score)

   print(f"AUROC: {metric.get():.3f}")

.. code-block:: text

   AUROC: 0.960

The metric gets each score while the stream runs. The detector never sees a label. As a result, the detector works under the same conditions as in a live system. This score-then-learn order is called prequential evaluation. :doc:`concepts` explains it.

Raise alerts
------------

Raw scores have no fixed scale. A score of 2.5 can be high for one detector and low for another. A calibrator converts each score into the fraction of recent scores that it exceeds. Use this calibrated value to decide when to raise an alert.

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

   print(f"{len(alerts)} alerts, {y[alerts].sum()} of them true anomalies")

.. code-block:: text

   20 alerts, 15 of them true anomalies

A calibrated value of 0.99 means that the score is higher than 99% of the last 500 scores. This alert threshold has the same meaning for every detector. The loop raises no alerts for the first 200 points (a short warm-up). The detector and the calibrator need some data before their output is stable.

After the warm-up, the stream has 36 anomalies, so at this threshold the detector finds 15 of them. To find more anomalies, lower the threshold, and expect more false alerts. :doc:`guides/evaluate` shows how to measure this trade-off with precision and recall.

Try real data
-------------

:class:`~pysad.utils.Data` gives you 17 labeled benchmark datasets (standard datasets for comparing detectors). It downloads each dataset the first time that you use it. Then it keeps a copy in ``~/.cache/pysad``.

.. code-block:: python

   from pysad.utils import Data

   X_cardio, y_cardio = Data().get_data("cardio.mat")

   model = LODA(random_state=0)
   metric = AUROCMetric()
   for x, label in streamer.iter(X_cardio, y_cardio):
       metric.update(label, model.fit_score_partial(x))

   print(f"AUROC on cardio: {metric.get():.3f}")

.. code-block:: text

   AUROC on cardio: 0.946

You built the full workflow: stream the data, score each point, measure the detector and raise alerts. The same loop works with every detector in PySAD.

Next steps
----------

- :doc:`guides/choose_detector` helps you choose a detector for your data.
- :doc:`guides/stream_data` connects PySAD to CSV files, DataFrames and message queues.
- :doc:`concepts` explains the streaming model, prequential evaluation and calibration.
- :doc:`api` lists every class and parameter.
