Calibration
===========

.. module:: pysad.transform.probability_calibration

This module holds probability calibrators. A calibrator wraps a detector and converts each raw score into a value between 0 and 1. Higher values mean more anomalous. Every calibrator here sets ``output_kind`` to ``"probability"``.

Raw score scales differ across detectors, so a value such as 2.5 can be normal for one detector and anomalous for another. A calibrator ranks each score against recent scores, so one alert threshold works across detectors. Chain a detector and a calibrator with ``|``:

.. code-block:: python

   import numpy as np

   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
   from pysad.utils import ArrayStreamer

   rng = np.random.default_rng(0)
   X = rng.normal(size=(2000, 4))
   streamer = ArrayStreamer()

   model = LODA()
   calibrator = ConformalProbabilityCalibrator(window_size=1000)
   pipeline = model | calibrator

   for t, x in enumerate(streamer.iter(X)):
       calibrated = pipeline.fit_score_partial(x)
       if calibrated >= 0.99:
           print(f"alert at t={t}")

``fit_score_partial`` scores each point with the current state, then learns from it. ``transform_partial`` reads the current state without changing it.

Streaming calibrators
---------------------

These calibrators fit a distribution to the scores they see:

- :class:`GaussianTailProbabilityCalibrator` fits a Gaussian to a window of recent scores and returns one minus the Gaussian tail of the short-term average. Use it when scores look bell-shaped.
- :class:`AnomalyLikelihoodProbabilityCalibrator` is the anomaly likelihood of HTM, the detector Numenta uses in NAB. It fits a Gaussian to short-term averages of scores.
- :class:`SPOTProbabilityCalibrator` fits a Generalized Pareto distribution to the highest scores. It makes no assumption about the score distribution. Set the risk ``q``, the probability that a normal score raises an alert.

Conformal calibrators
---------------------

These calibrators return one minus a conformal p-value: the fraction of recent scores below the input score.

- :class:`ConformalProbabilityCalibrator` keeps one sliding window of scores. It follows the sliding calibration queue of Ishimtsev et al. (2017).
- :class:`ConformalTSFMCalibrator` runs Adaptive Conformal Inference (Gibbs and Candès, 2021). It adapts the target miscoverage level from the miscoverage events it observes, and offers a momentum variant. Use it when the score distribution drifts.
- :class:`GroupConditionalConformalCalibrator` keeps one window per regime, for example an operating mode or a drift segment. It bounds the number of tracked regimes and evicts the least recently used one.

All calibrators return 0.0 before they have enough scores. Alert when the calibrated score exceeds ``1 - alpha`` for your target miscoverage ``alpha``, for example 0.99 for alpha 0.01.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated
