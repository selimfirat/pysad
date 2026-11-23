.. meta::
   :description: Normalize forecast residuals with TSFMResidualAdapter and calibrate them with ConformalTSFMCalibrator, then raise alerts on the calibrated scores.

Calibrate forecast residuals
============================

A forecaster predicts each value of a univariate stream. The absolute error of the forecast is the residual. :class:`~pysad.integrations.TSFMResidualAdapter` normalizes the residual with a rolling mean and standard deviation. :class:`~pysad.transform.probability_calibration.ConformalTSFMCalibrator` then converts the normalized residual into a calibrated score between 0 and 1. Higher values mean more anomalous.

The normalized residuals have a stable scale over time, which suits conformal calibration. Use this composition when you score a univariate stream by its forecast errors.

Build the forecaster
--------------------

The detector needs a forecaster with ``forecast(horizon)`` and ``learn_one(y)``. This moving-average forecaster has no external dependency:

.. code-block:: python

   class MovingAverageForecaster:
       """Predicts the next value as the mean of the last ``window`` values."""

       def __init__(self, window: int = 24) -> None:
           self.window = window
           self.history: list[float] = []

       def forecast(self, horizon: int = 1) -> list[float]:
           if not self.history:
               return [0.0]
           recent = self.history[-self.window:]
           return [sum(recent) / len(recent)]

       def learn_one(self, y: float) -> None:
           self.history.append(float(y))

Build the pipeline
------------------

Chain the detector, the residual adapter and the calibrator with the ``|`` operator:

.. code-block:: python

   from pysad.integrations import ForecastingAnomalyDetector, TSFMResidualAdapter
   from pysad.transform.probability_calibration import ConformalTSFMCalibrator

   forecaster = MovingAverageForecaster(window=24)
   detector = ForecastingAnomalyDetector(forecaster, probationary_period=50)
   adapter = TSFMResidualAdapter(window_size=100)
   calibrator = ConformalTSFMCalibrator(alpha=0.1, gamma=0.01, window_size=200)

   pipeline = detector | adapter | calibrator

``ForecastingAnomalyDetector`` scores each value by its absolute forecast error. ``TSFMResidualAdapter`` normalizes that error. ``ConformalTSFMCalibrator`` turns the normalized error into a calibrated score. The pipeline outputs a probability, so it supports ``predict_partial`` only after a decision stage such as :class:`~pysad.transform.postprocessing.ThresholdFilter`.

Score and calibrate each point
------------------------------

Stream the values one at a time and score each point before the stages learn from it:

.. code-block:: python

   import numpy as np

   from pysad.utils import ArrayStreamer

   rng = np.random.default_rng(0)
   t = np.arange(2000)
   values = np.sin(2 * np.pi * t / 24) + 0.1 * rng.normal(size=t.size)
   spikes = rng.choice(np.arange(100, t.size), size=40, replace=False)
   values[spikes] += rng.choice([-1.0, 1.0], size=spikes.size)

   X = values.reshape(-1, 1)
   streamer = ArrayStreamer()

   alerts = []
   for index, x in enumerate(streamer.iter(X)):
       calibrated = pipeline.fit_score_partial(x)
       if index >= 200 and calibrated >= 0.99:
           alerts.append(index)

   print(f"{len(alerts)} alerts")

``fit_score_partial`` scores with the current state and then fits every stage, so the calibrator always transforms against past data only. Skip the first 200 points while the forecaster, the adapter and the calibrator initialize.

Raise alerts
------------

The calibrated score is a value between 0 and 1. A threshold of 0.99 flags about 1 in 100 normal points. Raise the threshold to reduce false alerts, or lower it to catch more anomalies. Measure the trade-off with precision and recall, as :doc:`/guides/evaluate` shows.

You can also run the stages in an explicit loop instead of the pipeline:

.. code-block:: python

   for index, x in enumerate(streamer.iter(X)):
       value = float(x[0])
       error = detector.score_partial(x)
       normalized = adapter.transform_partial(error)
       calibrated = calibrator.fit_transform_partial(normalized)
       detector.fit_partial(x)
       adapter.fit_partial(error)

Here ``detector.score_partial`` runs before ``detector.fit_partial``, matching the score-then-learn loop. The pipeline does the same order automatically.

Next steps
----------

- :doc:`/guides/alerts` explains thresholds and warm-up for calibrators.
- :doc:`/guides/evaluate` explains how to measure precision, recall and AUROC.
- :doc:`/guides/pipelines` explains how pipeline stages connect.
