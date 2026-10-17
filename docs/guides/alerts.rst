Turn scores into alerts
=======================

A detector gives a raw score with no fixed scale. To decide when to alert, calibrate the score to a value between 0 and 1. Then compare the calibrated value with a threshold.

The examples on this page use a synthetic stream. Replace ``X`` with your own data:

.. code-block:: python

   import numpy as np

   from pysad.utils import ArrayStreamer

   X = np.random.default_rng(42).normal(size=(2000, 4))
   streamer = ArrayStreamer()

Calibrate and alert
-------------------

:class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator` converts each score into the fraction of recent scores that are lower than that score:

.. code-block:: python

   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   model = LODA()
   calibrator = ConformalProbabilityCalibrator(window_size=1000)

   for t, x in enumerate(streamer.iter(X)):
       score = model.fit_score_partial(x)
       calibrated = calibrator.fit_transform_partial(score)
       if t >= 200 and calibrated >= 0.99:
           print(f"alert at t={t}")

The condition ``t >= 200`` skips a warm-up period. During the warm-up, the model and the calibrator have too little data, and their output is not stable.

Choose the threshold
--------------------

The threshold sets how often a normal point raises an alert:

.. list-table::
   :header-rows: 1

   * - Threshold
     - Normal points that raise an alert
   * - 0.95
     - About 1 in 20
   * - 0.99
     - About 1 in 100
   * - 0.999
     - About 1 in 1,000

To get fewer false alerts, increase the threshold. The calibrator window must hold enough scores for the threshold. With 1,000 scores, only the highest score in the window reaches 0.999, so for a threshold of 0.999, use a ``window_size`` of 5,000 or more.

The ``window_size`` of the calibrator also sets what "recent" means. A small window follows changes in the data faster. A large window gives a more stable threshold.

Calibrate with a Gaussian fit
-----------------------------

:class:`~pysad.transform.probability_calibration.GaussianTailProbabilityCalibrator` keeps a running mean and variance of the scores over a window:

.. code-block:: python

   from pysad.transform.probability_calibration import GaussianTailProbabilityCalibrator

   calibrator = GaussianTailProbabilityCalibrator(window_size=1000)

It assumes that the scores are approximately Gaussian (bell-shaped). Many detectors give skewed scores, with a long tail of high values. Then this calibrator raises more alerts than the threshold suggests. If you are not sure, use the conformal calibrator.

Smooth noisy scores
-------------------

If single noisy points raise alerts, average the scores over a short window before you calibrate them:

.. code-block:: python

   from pysad.transform.postprocessing import RunningAveragePostprocessor

   smoother = RunningAveragePostprocessor(window_size=5)

   for x in streamer.iter(X):
       score = model.fit_score_partial(x)
       smoothed = smoother.fit_transform_partial(score)
       calibrated = calibrator.fit_transform_partial(smoothed)

Smoothing delays the alerts and hides anomalies that last for only one point. Use it for anomalies that last for several points, such as a level shift (a lasting jump in the value).

Keep anomalies out of the model
-------------------------------

With ``fit_score_partial``, the model learns from every point, including anomalies. A long anomaly can therefore become normal for the model. To prevent this, score each point first and learn from it only if it does not raise an alert:

.. code-block:: python

   alerts = []
   for t, x in enumerate(streamer.iter(X)):
       if t < 200:  # warm-up: learn from every point
           calibrator.fit_partial(model.fit_score_partial(x))
           continue

       score = model.score_partial(x)
       if calibrator.fit_transform_partial(score) >= 0.99:
           alerts.append(t)  # alert, and keep the point out of the model
       else:
           model.fit_partial(x)

If the data changes permanently, for example after a deployment, this model never learns the new normal behavior. After you confirm the change, reset the model or let it learn from all points again.

Next steps
----------

- :doc:`evaluate` measures how well the alerts work.
- :doc:`production` covers warm start, saved state and latency.
