Turn scores into alerts
=======================

Calibrate raw scores to a 0-to-1 scale, then compare them with an alert threshold.

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

``t >= 200`` suppresses alerts during the first 200 points while the model and calibrator initialize.

:class:`~pysad.transform.postprocessing.ThresholdFilter` produces 0/1 labels for metrics such as :class:`~pysad.evaluation.PrecisionMetric`: 1 at or above the threshold, 0 otherwise:

.. code-block:: python

   from pysad.transform.postprocessing import ThresholdFilter

   threshold_filter = ThresholdFilter(threshold=0.99)
   label = threshold_filter.transform_partial(calibrated)

Choose the threshold
--------------------

These approximate rates assume stable normal scores with few ties. Drift, dependence and anomalies in the window affect the rates; validate your threshold on labeled data using :doc:`evaluate`.

.. list-table::
   :header-rows: 1

   * - Threshold
     - Approximate normal-point alert rate
   * - 0.95
     - About 1 in 20
   * - 0.99
     - About 1 in 100
   * - 0.999
     - About 1 in 1,000

Raise the threshold to reduce false alerts on normal points. With 1,000 scores, only the highest reaches 0.999. For a 0.999 threshold, use ``window_size`` of 5,000 or more.

``window_size`` sets the calibration history: smaller windows adapt faster; larger windows give more stable ranks.

Calibrate with a Gaussian fit
-----------------------------

:class:`~pysad.transform.probability_calibration.GaussianTailProbabilityCalibrator` updates a windowed mean and variance:

.. code-block:: python

   from pysad.transform.probability_calibration import GaussianTailProbabilityCalibrator

   calibrator = GaussianTailProbabilityCalibrator(window_size=1000)

The Gaussian fit assumes approximately bell-shaped scores. Skewed scores with a long upper tail can raise more alerts than expected. Use conformal calibration if this assumption is uncertain.

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

Scores above ``calibrator.threshold_`` map above ``1 - q`` and are excluded from fitting. The fitted tail extends beyond observed scores, so small ``q`` values do not require large windows. The first ``num_init`` scores return 0.

For drifting scores, set ``depth``, for example ``depth=10``, to fit differences from the mean of the last ``depth`` scores.

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

Label the scores with a running quantile
----------------------------------------

:class:`~pysad.transform.postprocessing.QuantileFilter` gives a score the label 1 if the score is at least the ``q`` quantile of the previous scores, and 0 otherwise. It keeps an estimate of the quantile in constant memory, so it uses every score since the start of the stream. This is the ``QuantileFilter`` of River:

.. code-block:: python

   from pysad.transform.postprocessing import QuantileFilter

   quantile_filter = QuantileFilter(q=0.99)

   alerts = []
   for t, x in enumerate(streamer.iter(X)):
       if quantile_filter.fit_transform_partial(model.fit_score_partial(x)):
           alerts.append(t)

For stable scores with few ties, about 1 in 100 gets label 1 even without anomalies. The filter retains old information, so it adapts slowly to score changes.

Smooth noisy scores
-------------------

If single noisy points raise alerts, average the scores over a short window before you calibrate them:

.. code-block:: python

   from pysad.transform.postprocessing import AveragePostprocessor

   model = LODA()
   smoother = AveragePostprocessor(window_size=5)
   calibrator = ConformalProbabilityCalibrator(window_size=1000)

   for x in streamer.iter(X):
       score = model.fit_score_partial(x)
       smoothed = smoother.fit_transform_partial(score)
       calibrated = calibrator.fit_transform_partial(smoothed)

Smoothing can delay alerts and hide isolated anomalies. Use it for sustained anomalies, such as level shifts.

Keep anomalies out of the model
-------------------------------

``fit_score_partial`` learns from anomalies too, so sustained anomalies can become normal to the model. To exclude flagged points, score first and learn only when no alert is raised:

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

This filter can prevent learning a permanent change, such as a deployment. If that happens, reset the model or resume learning from all points.

Plot the alerts
---------------

``examples/example_plot_stream.py`` in :doc:`../examples` plots features, calibrated scores with the threshold, and alerts against labels. Install matplotlib separately with ``pip install matplotlib``.

Next steps
----------

- :doc:`evaluate` shows how to measure how well the alerts work.
- :doc:`production` covers warm start, saved state and latency.
