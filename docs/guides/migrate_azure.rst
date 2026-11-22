.. meta::
   :description: Migrate from retired Azure AI Anomaly Detector and Azure Metrics Advisor to PySAD in Python with SpectralResidual, SeasonalHybridESD, LODA, and calibrated streaming alerts.

Migrate from Azure AI Anomaly Detector and Metrics Advisor
============================================================

Microsoft retired Azure AI Anomaly Detector on October 1, 2026. If your pipelines call the univariate entire/last endpoints, the multivariate API, or Azure Metrics Advisor feeds built on the same Spectral Residual and STL families, PySAD gives you self-hosted Python equivalents under the BSD 3-Clause license.

Scope and operational differences
---------------------------------

PySAD is an in-process Python library, not a managed cloud service:

- **No hosted REST API or scheduling engine**: Run the detectors inside your own Kafka/Flink worker or microservice (see :doc:`serving`).
- **One observation at a time**: Rather than re-sending overlapping historical arrays per HTTP request, PySAD maintains bounded rolling state in memory across ``fit_score_partial`` calls.
- **No severity margin bands or root-cause UI**: Detectors emit an anomaly score (or binary ``0`` / ``1`` alert via ``predict_partial``), not ``expectedValue``, ``upperMargin``, or multi-dimension incident trees. If you need forecast deviations on seasonal telemetry, pair :class:`~pysad.models.HoltWinters` or :class:`~pysad.integrations.PredictiveAnomalyDetector`.

API mapping
-----------

.. list-table::
   :header-rows: 1
   :widths: 34 36 30

   * - Azure capability
     - PySAD replacement
     - Notes
   * - Univariate ``detect_last_point`` / ``detect_entire_series`` (non-seasonal)
     - :class:`~pysad.models.SpectralResidual`
     - Port of Microsoft's SR algorithm from the Azure paper.
   * - Univariate ``period`` / seasonal detection
     - :class:`~pysad.models.SeasonalHybridESD`
     - Modified STL decomposition + generalized ESD test; positive scores flag anomalies.
   * - Custom ``sensitivity`` / ``maxAnomalyRatio``
     - :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator` + :class:`~pysad.transform.postprocessing.ThresholdFilter`
     - Maps raw scores to empirical tail probabilities in ``[0, 1]`` for thresholding.
   * - Tail alert tuning with drift
     - :class:`~pysad.transform.probability_calibration.SPOTProbabilityCalibrator`
     - Extreme-value threshold (``q`` risk level; set ``depth > 0`` for DSPOT).
   * - Multivariate Anomaly Detection (``MultivariateClient``)
     - :class:`~pysad.models.OnlineIsolationForest`, :class:`~pysad.models.LODA`, or :class:`~pysad.models.RobustRandomCutForest`
     - Scales across correlated sensor vectors with bounded memory.
   * - ``detect_change_point``
     - :class:`~pysad.statistics.ADWINMeter`, :class:`~pysad.statistics.KSWINMeter`, or :class:`~pysad.statistics.PageHinkleyMeter`
     - Detects mean or distribution shifts online without storing the full stream.
   * - Multi-metric Metrics Advisor feeds
     - Dictionary of pipelines keyed by ``(metric, dimension_key)``
     - Keep per-series rolling windows isolated in Python or Redis/stateful workers.

Replace univariate ``detect_last_point``
----------------------------------------

Azure's univariate API splits workloads into non-seasonal Spectral Residual and periodic STL/ESD checks. You can keep Spectral Residual directly in Python and convert its saliency scores into calibrated binary alerts, or use :class:`~pysad.models.SeasonalHybridESD` for hourly or daily cycles (see also :doc:`seasonal`):

.. code-block:: python

   import numpy as np

   from pysad.models import SpectralResidual
   from pysad.transform.postprocessing import ThresholdFilter
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   rng = np.random.default_rng(0)
   signal = 50.0 + rng.normal(scale=0.3, size=250)
   signal[210:215] = 65.0  # injected anomaly burst

   pipeline = (
       SpectralResidual(window_size=64)
       | ConformalProbabilityCalibrator(window_size=200)
       | ThresholdFilter(threshold=0.99)
   )

   for step, value in enumerate(signal):
       alert = pipeline.fit_predict_partial(np.array([value]))
       if alert == 1 and step >= 100:
           print(f"anomaly at step {step}: value={value:.2f}")

Replace multivariate anomaly detection
--------------------------------------

Azure's multivariate endpoint learns correlations across multiple numeric time series. In PySAD, normalize features online with :class:`~pysad.transform.preprocessing.StandardScaler`, score with a bounded multivariate detector, and attach a calibrator and threshold:

.. code-block:: python

   import numpy as np

   from pysad.models import OnlineIsolationForest
   from pysad.transform.postprocessing import ThresholdFilter
   from pysad.transform.preprocessing import StandardScaler
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   rng = np.random.default_rng(42)
   normal = rng.normal(loc=0.0, scale=1.0, size=(300, 4))
   normal[260] = [6.0, 6.0, -6.0, -6.0]  # cross-sensor correlation break

   pipeline = (
       StandardScaler()
       | OnlineIsolationForest(num_trees=50, window_size=256, random_state=42)
       | ConformalProbabilityCalibrator(window_size=200)
       | ThresholdFilter(threshold=0.99)
   )

   for step, x in enumerate(normal):
       alert = pipeline.fit_predict_partial(x)
       if alert == 1 and step >= 150:
           print(f"multivariate alert at step {step}")

Next steps
----------

- Tune alert sensitivity, smoothing, and conformal p-values in :doc:`alerts`.
- Run your pipelines behind FastAPI or a stream consumer in :doc:`serving`.
- Handle hourly or weekly cycles explicitly in :doc:`seasonal`.
