Getting started
===============

We will use LODA to score a synthetic stream, evaluate the scores, and raise alerts. Allow ten minutes and run the code blocks in order in one Python session.

You need Python 3.10 or newer. You do not need to know about anomaly detection.

Install PySAD
-------------

Install PySAD with pip:

.. code-block:: console

   $ pip install pysad

For conda and other installation options, see :doc:`installation`.

Make a stream
-------------

Generate a stream with known anomalies. Each row is a NumPy feature array; we will process the rows in order.

The stream has 2,000 points with 4 features. Every 50th point is an anomaly: the code adds 3 to its first two features.

.. code-block:: python

   import numpy as np

   rng = np.random.default_rng(42)
   X = rng.normal(size=(2000, 4))
   y = np.zeros(2000, dtype=int)
   y[::50] = 1
   X[y == 1, :2] += 3

``X`` holds the points; ``y`` holds labels: 1 for an anomaly, 0 for normal. Only the evaluation uses ``y``.

Score each point
----------------

Create a :class:`~pysad.models.LODA` detector and feed it one point at a time.

.. code-block:: python

   from pysad.models import LODA
   from pysad.utils import ArrayStreamer

   model = LODA(random_state=0)
   streamer = ArrayStreamer()

   scores = []
   for x in streamer.iter(X):
       score = model.fit_score_partial(x)
       scores.append(score)

:class:`~pysad.utils.ArrayStreamer` yields one row at a time. :meth:`~pysad.core.BaseModel.fit_score_partial` scores each point before learning from it. Scores are Python ``float`` values; higher means more anomalous.

Compare the average score of the normal points and of the anomalies:

.. code-block:: python

   scores = np.array(scores)
   print(f"normal: {scores[y == 0].mean():.2f}")
   print(f"anomalies: {scores[y == 1].mean():.2f}")

.. code-block:: text

   normal: 1.50
   anomalies: 2.63

The anomalies have a higher average score, although the detector learned without labels.

Measure the detector
--------------------

We will measure the ranking with AUROC (the area under the ROC curve). A perfect ranking gets 1.0; random scores get 0.5 on average. See :doc:`guides/evaluate` for metric definitions and choices.

Update the metric after each score:

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

Scoring before learning measures performance on unseen points. This is prequential evaluation; see :doc:`concepts`.

Raise alerts
------------

Raw score scales differ: 2.5 can be high for one detector and low for another. Conformal calibration ranks each score against recent scores so we can set an alert threshold.

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

A value of 0.99 means that 99% of the window’s scores are lower. The window holds up to 500 scores, including the current score. This rank is not an anomaly probability. We skip alerts for a 200-point warm-up; choose a suitable warm-up, threshold and window for your data using :doc:`guides/alerts`.

After warm-up, the detector finds 15 of 36 anomalies. Lowering the threshold can find more anomalies but also raises more false alerts. Measure this trade-off with precision and recall; see :doc:`guides/evaluate`.

Try real data
-------------

:class:`~pysad.utils.Data` provides |num_datasets| labeled benchmark datasets. It downloads each on first use and caches it in ``~/.cache/pysad``.

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

The same stream, score, evaluate and alert workflow works with every PySAD detector.

Next steps
----------

- :doc:`guides/choose_detector` helps you choose a detector for your data.
- :doc:`guides/stream_data` connects PySAD to CSV files, DataFrames and message queues.
- :doc:`concepts` explains the streaming model, prequential evaluation and calibration.
- :doc:`api` lists every class and parameter.
