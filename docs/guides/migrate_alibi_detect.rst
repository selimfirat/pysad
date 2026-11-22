.. meta::
   :description: Migrate from BSL 1.1 Seldon Alibi Detect to BSD 3-Clause PySAD for commercial-friendly Python streaming outlier detection, concept drift monitoring, and sliding-window PyOD wrappers.

Migrate from Alibi Detect (BSL 1.1) to PySAD (BSD 3-Clause)
===========================================================

Seldon relicensed ``alibi-detect`` under the Business Source License (BSL 1.1), restricting commercial production and managed-service deployments without a paid license. PySAD is published under the permissive **BSD 3-Clause** license, supports Python 3.11+ and NumPy 2, and covers online outlier detection, probability calibration, concept drift detection, and batch model windowing. For a side-by-side feature breakdown, see :doc:`../comparisons/alibi_detect`.

Scope and what belongs elsewhere
--------------------------------

PySAD is built for **tabular, time-series, and embedding vector streams**:

- **No vision/image adversarial or PyTorch/TF generative architectures**: PySAD does not bundle CNN autoencoders, VAE/GAN image outlier detectors, or pixel-space adversarial detectors. If your pipeline detects outliers over pretrained neural embeddings (text, audio, or vision encoders), extract vectors in :class:`~pysad.transform.preprocessing.FunctionTransformer` and score the embeddings in PySAD.
- **Streaming-first memory**: PySAD models update point-by-point in bounded space; use :class:`~pysad.integrations.OneFitModel` or :class:`~pysad.integrations.ReferenceWindowModel` when migrating a batch detector (PyOD or scikit-learn) to streams.

API mapping
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Alibi Detect detector
     - PySAD equivalent
     - Notes
   * - ``Mahalanobis`` (online tabular)
     - :class:`~pysad.models.OnlineIsolationForest`, :class:`~pysad.models.LODA`, or :class:`~pysad.models.ExactStorm`
     - Stateful online detectors with bounded memory.
   * - ``SpectralResidual``
     - :class:`~pysad.models.SpectralResidual`
     - Univariate spectral saliency detector.
   * - ``IForest`` / batch ``OutlierAE`` equivalents
     - :class:`~pysad.integrations.OneFitModel` or :class:`~pysad.integrations.ReferenceWindowModel`
     - Wraps any scikit-learn or PyOD detector over initial or sliding windows (see :doc:`pyod`).
   * - ``KSDriftOnline`` / ``MMDDriftOnline`` (stream drift)
     - :class:`~pysad.statistics.KSWINMeter`, :class:`~pysad.statistics.ADWINMeter`, :class:`~pysad.statistics.PageHinkleyMeter`
     - Univariate / score-stream drift statistics in :mod:`pysad.statistics`.
   * - Detector auto-reset on concept drift
     - :class:`~pysad.models.DriftAdaptiveDetector`
     - Monitors detector output scores with :class:`~pysad.statistics.PageHinkleyMeter` and rebuilds on shift.
   * - Threshold calibration (``infer_threshold``)
     - :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator` or :class:`~pysad.transform.probability_calibration.SPOTProbabilityCalibrator`
     - Online distribution-free p-values and extreme-value tails replacing offline holdout thresholds.

Replace online tabular outlier detection and calibration
--------------------------------------------------------

Instead of calibrating thresholds offline on a reference split with ``infer_threshold``, chain online feature scaling, a streaming detector, conformal calibration, and a probability cutoff:

.. code-block:: python

   import numpy as np

   from pysad.models import OnlineIsolationForest
   from pysad.transform.postprocessing import ThresholdFilter
   from pysad.transform.preprocessing import StandardScaler
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   rng = np.random.default_rng(5)
   stream = rng.normal(loc=0.0, scale=1.0, size=(300, 6))
   stream[270] += 6.0  # out-of-distribution tabular sample

   pipeline = (
       StandardScaler()
       | OnlineIsolationForest(num_trees=50, window_size=256, random_state=5)
       | ConformalProbabilityCalibrator(window_size=200)
       | ThresholdFilter(threshold=0.98)
   )

   for step, x in enumerate(stream):
       alert = pipeline.fit_predict_partial(x)
       if alert == 1 and step >= 150:
           print(f"outlier alert at step {step}")

Replace online concept drift monitoring and adaptation
------------------------------------------------------

PySAD includes ten streaming drift meters (``ADWINMeter``, ``KSWINMeter``, ``PageHinkleyMeter``, ``DDMMeter``, and related detectors) and :class:`~pysad.models.DriftAdaptiveDetector` to reset an anomaly detector automatically when incoming data shifts:

.. code-block:: python

   import numpy as np

   from pysad.models import DriftAdaptiveDetector, LODA
   from pysad.statistics import KSWINMeter, PageHinkleyMeter

   rng = np.random.default_rng(9)
   kswin = KSWINMeter(alpha=0.005, window_size=200, stat_size=30, random_state=9)
   adaptive = DriftAdaptiveDetector(
       model_factory=lambda: LODA(random_state=9),
       drift_meter=PageHinkleyMeter(delta=0.005, threshold=25.0),
   )

   before = rng.normal(loc=0.0, scale=1.0, size=(200, 4))
   after = rng.normal(loc=5.0, scale=1.0, size=(120, 4))

   for step, x in enumerate(np.vstack([before, after])):
       score = adaptive.fit_score_partial(x)
       kswin.update(float(x[0]))
       if kswin.drift_detected:
           print(f"feature drift detected at step {step} (score={score:.3f})")

Next steps
----------

- Compare supported features across open-source detectors in :doc:`../comparisons/alibi_detect`.
- Run batch PyOD and scikit-learn detectors on sliding windows in :doc:`pyod`.
- Calibrate alert thresholds without offline holdouts in :doc:`alerts`.
