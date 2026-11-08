Turn scores into alerts
=======================

A detector gives a raw score with no fixed scale. To decide when to raise an alert, first calibrate the score. Calibration maps the score to a value between 0 and 1. Then compare the calibrated value with a threshold.

The examples on this page use a synthetic stream. Replace ``X`` with your own data:

.. code-block:: python

   import numpy as np

   from pysad.utils import ArrayStreamer

   X = np.random.default_rng(42).normal(size=(2000, 4))
   streamer = ArrayStreamer()

Calibrate the scores and raise alerts
-------------------------------------

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

The condition ``t >= 200`` skips a warm-up period of 200 points. During the warm-up, the model and the calibrator have too little data, and their output is not stable.

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

To get fewer false alerts (alerts for normal points), increase the threshold. The window of the calibrator must hold enough scores for the threshold. With 1,000 scores, only the highest score in the window reaches 0.999. If the threshold is 0.999, use a ``window_size`` of 5,000 or more.

The ``window_size`` of the calibrator also sets what "recent" means. A small window follows changes in the data faster. A large window gives a more stable threshold.

Calibrate with a Gaussian fit
-----------------------------

:class:`~pysad.transform.probability_calibration.GaussianTailProbabilityCalibrator` keeps the mean and the variance of the scores in a window, and updates them with each score:

.. code-block:: python

   from pysad.transform.probability_calibration import GaussianTailProbabilityCalibrator

   calibrator = GaussianTailProbabilityCalibrator(window_size=1000)

This calibrator assumes that the scores are approximately Gaussian (bell-shaped). Many detectors give skewed scores, with a long tail of high values. For skewed scores, this calibrator raises more alerts than the threshold suggests. If you are not sure that your scores are Gaussian, use the conformal calibrator.

Calibrate with extreme value theory
-----------------------------------

:class:`~pysad.transform.probability_calibration.SPOTProbabilityCalibrator` fits a Generalized Pareto distribution to the scores above a high quantile of the first ``num_init`` scores. You set the risk ``q``, which is the probability that a normal score raises an alert:

.. code-block:: python

   from pysad.transform.probability_calibration import SPOTProbabilityCalibrator

   calibrator = SPOTProbabilityCalibrator(q=1e-3, num_init=500)

   alerts = []
   for t, x in enumerate(streamer.iter(X)):
       if calibrator.fit_transform_partial(model.fit_score_partial(x)) > 1 - 1e-3:
           alerts.append(t)

A score gets a calibrated value above ``1 - q`` when the score is above ``calibrator.threshold_``. The calibrator does not learn from these scores. Unlike the conformal calibrator, this calibrator does not need a large window for a small ``q``, because the fitted distribution extends beyond the highest score seen so far. It returns 0 for the first ``num_init`` scores.

If the scores drift, set ``depth``, for example ``depth=10``. The calibrator then fits the distribution to the difference between each score and the mean of the last ``depth`` scores.

Calibrate with the anomaly likelihood of HTM
--------------------------------------------

:class:`~pysad.transform.probability_calibration.AnomalyLikelihoodProbabilityCalibrator` gives the anomaly likelihood that the HTM detector of Numenta uses in NAB. It averages the last 10 scores, fits a Gaussian to these averages, and gives a value near 1 when the newest average is far from their mean. A single high score changes the average only a little, so the calibrator reacts to a sustained increase of the scores more than to one noisy point. Its bounds on the fitted Gaussian expect scores between 0 and 1. The paper alerts at a likelihood of at least ``1 - 1e-5``:

.. code-block:: python

   from pysad.transform.probability_calibration import AnomalyLikelihoodProbabilityCalibrator

   calibrator = AnomalyLikelihoodProbabilityCalibrator()

   alerts = []
   for t, x in enumerate(streamer.iter(X)):
       if calibrator.fit_transform_partial(model.fit_score_partial(x)) >= 1 - 1e-5:
           alerts.append(t)

The calibrator returns 0.5 for the first ``learning_period + num_estimation_samples`` scores, 388 by default. During a run of anomalous scores, only the first score reaches ``1 - 1e-5``. The next scores of the run get ``0.999``.

Threshold the scores of a window without a risk level
-----------------------------------------------------

:class:`~pysad.transform.postprocessing.NonparametricDynamicThresholdPostprocessor` is the threshold of Telemanom. It smooths the scores and picks the threshold that removes the most of the mean and the standard deviation of the recent scores with the fewest anomalies. Then it drops the anomalous sequences whose maximum is close to the next lower maximum. The output is 0 for a normal point, so an alert is a score above 0. The scores must be nonnegative, for example prediction errors:

.. code-block:: python

   from pysad.transform.postprocessing import NonparametricDynamicThresholdPostprocessor

   threshold = NonparametricDynamicThresholdPostprocessor()

   alerts = []
   for t, x in enumerate(streamer.iter(X)):
       if threshold.fit_transform_partial(model.fit_score_partial(x)) > 0:
           alerts.append(t)

Smooth noisy scores
-------------------

If single noisy points raise alerts, average the scores over a short window before you calibrate them:

.. code-block:: python

   from pysad.transform.postprocessing import RunningAveragePostprocessor

   model = LODA()
   smoother = RunningAveragePostprocessor(window_size=5)
   calibrator = ConformalProbabilityCalibrator(window_size=1000)

   for x in streamer.iter(X):
       score = model.fit_score_partial(x)
       smoothed = smoother.fit_transform_partial(score)
       calibrated = calibrator.fit_transform_partial(smoothed)

Smoothing delays the alerts and hides anomalies that last for only one point. Use smoothing for anomalies that last for several points, for example a level shift (a lasting jump in the value).

Keep anomalies out of the model
-------------------------------

With ``fit_score_partial``, the model learns from every point, and also from anomalies. As a result, a long anomaly can become normal for the model. To prevent this, score each point first. Then learn from the point only if it does not raise an alert:

.. code-block:: python

   model = LODA()
   calibrator = ConformalProbabilityCalibrator(window_size=1000)

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

If the data changes permanently, for example after a deployment, the model in this loop never learns the new normal behavior. If you find such a change, reset the model, or let it learn from all points again.

Plot the alerts
---------------

To see where the alerts fall, collect the calibrated scores and pass them to :func:`~pysad.utils.plot_stream` with the threshold. The function needs matplotlib, so install it with ``pip install matplotlib``. The figure has three panels: the features, the scores with the threshold, and the alerts. If you have labels, pass them as ``y`` to show the anomalies under the alerts:

.. code-block:: python

   from pysad.utils import plot_stream

   model = LODA()
   calibrator = ConformalProbabilityCalibrator(window_size=1000)

   calibrated = [calibrator.fit_transform_partial(model.fit_score_partial(x)) for x in streamer.iter(X)]
   plot_stream(X, np.array(calibrated), threshold=0.99)

For :class:`~pysad.transform.probability_calibration.SPOTProbabilityCalibrator`, pass the raw scores and an array with ``threshold_`` after each point.

Next steps
----------

- :doc:`evaluate` shows how to measure how well the alerts work.
- :doc:`production` covers warm start, saved state and latency.
