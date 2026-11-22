.. meta::
   :description: Replace retired AWS Lookout for Equipment and Kinesis RANDOM_CUT_FOREST in Python with PySAD RobustRandomCutForest, ShingleTransformer, and OnlineIsolationForest.

Migrate from AWS Lookout for Equipment and Kinesis RCF
======================================================

AWS retired Amazon Lookout for Equipment on October 7, 2026. Meanwhile, teams moving workloads off Amazon Kinesis Data Analytics SQL (which uses the ``RANDOM_CUT_FOREST`` and ``RANDOM_CUT_FOREST_WITH_EXPLANATION`` functions based on Guha et al., 2016) need an in-process Python replacement that preserves shingling, tree decay, and bounded streaming memory.

Scope and operational differences
---------------------------------

PySAD is an open-source Python library (BSD 3-Clause), not a managed AWS service:

- **No managed S3 training scheduler or fleet control plane**: Execute PySAD inside AWS Lambda, ECS, PyFlink, or a microservice (see :doc:`serving`).
- **Continuous streaming updates**: Instead of training separate batch jobs in Lookout for Equipment and exporting scheduled CSV predictions, PySAD models update incrementally on each incoming telemetry vector (or wrap historical normal batches via :doc:`pyod`).
- **Shingled RRCF or explicit preprocessing**: PySAD includes the original Guha et al. (2016) :class:`~pysad.models.RobustRandomCutForest` algorithm directly, alongside :class:`~pysad.transform.preprocessing.ShingleTransformer` for pairing sliding context with any multivariate detector.

API mapping
-----------

.. list-table::
   :header-rows: 1
   :widths: 36 34 30

   * - AWS concept
     - PySAD replacement
     - Notes
   * - Kinesis ``RANDOM_CUT_FOREST`` (``numberOfTrees``, ``subSampleSize``, ``timeDecay``, ``shingleSize``)
     - :class:`~pysad.models.RobustRandomCutForest`
     - Parameters ``num_trees``, ``tree_size``, ``time_decay``, ``shingle_size``, and ``random_state``.
   * - Kinesis ``RANDOM_CUT_FOREST_WITH_EXPLANATION`` (feature attribution)
     - :meth:`HalfSpaceTrees.explain_partial <pysad.models.HalfSpaceTrees.explain_partial>`
     - Returns per-feature contributions alongside Half-Space Trees scores.
   * - Lookout for Equipment historical normal training + online scoring
     - :class:`~pysad.transform.preprocessing.StandardScaler` + :class:`~pysad.transform.preprocessing.ShingleTransformer` + :class:`~pysad.models.OnlineIsolationForest`
     - Bounded sliding window with online feature scaling and temporal context.
   * - Retraining an offline equipment model periodically
     - :class:`~pysad.integrations.OneFitModel` or :class:`~pysad.integrations.ReferenceWindowModel`
     - Fit PyOD or scikit-learn estimators on reference maintenance runs (see :doc:`pyod`).
   * - Equipment regime shifts / maintenance recalibration
     - :class:`~pysad.models.DriftAdaptiveDetector` + :class:`~pysad.statistics.ADWINMeter`
     - Resets the underlying detector when stream score statistics drift.
   * - Score thresholding / alert rate control
     - :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator` + :class:`~pysad.transform.postprocessing.ThresholdFilter`
     - Controls false alarm rate via sliding conformal quantiles.

Replace Kinesis ``RANDOM_CUT_FOREST``
-------------------------------------

A typical Kinesis SQL query runs ``RANDOM_CUT_FOREST`` with 100 trees, sample size 256, shingle size 4, and exponential decay. The direct PySAD equivalent uses :class:`~pysad.models.RobustRandomCutForest`:

.. code-block:: python

   import numpy as np

   from pysad.models import RobustRandomCutForest
   from pysad.transform.postprocessing import ThresholdFilter
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   rng = np.random.default_rng(7)
   t = np.arange(350)
   stream = np.sin(2 * np.pi * t / 24) + rng.normal(scale=0.1, size=t.size)
   stream[290:295] += 4.5  # injected waveform anomaly

   rcf_pipeline = (
       RobustRandomCutForest(
           num_trees=100,
           tree_size=256,
           shingle_size=4,
           time_decay=0.01,
           random_state=7,
       )
       | ConformalProbabilityCalibrator(window_size=200)
       | ThresholdFilter(threshold=0.98)
   )

   for step, value in enumerate(stream):
       alert = rcf_pipeline.fit_predict_partial(np.array([value]))
       if alert == 1 and step >= 200:
           print(f"RCF alert at step {step}: value={value:.2f}")

Replace Lookout for Equipment multivariate monitoring
-----------------------------------------------------

Industrial assets stream correlated vibration, temperature, and pressure readings. Combining :class:`~pysad.transform.preprocessing.StandardScaler` with :class:`~pysad.transform.preprocessing.ShingleTransformer` captures both cross-sensor relationships and short temporal dynamics:

.. code-block:: python

   import numpy as np

   from pysad.models import OnlineIsolationForest
   from pysad.transform.postprocessing import ThresholdFilter
   from pysad.transform.preprocessing import ShingleTransformer, StandardScaler
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   rng = np.random.default_rng(11)
   sensors = rng.normal(loc=0.0, scale=1.0, size=(320, 5))
   sensors[280:284, :3] += 6.0  # multi-step sensor fault

   equipment_pipeline = (
       StandardScaler()
       | ShingleTransformer(shingle_size=4)
       | OnlineIsolationForest(num_trees=50, window_size=256, random_state=11)
       | ConformalProbabilityCalibrator(window_size=200)
       | ThresholdFilter(threshold=0.98)
   )

   for step, reading in enumerate(sensors):
       alert = equipment_pipeline.fit_predict_partial(reading)
       if alert == 1 and step >= 200:
           print(f"equipment anomaly at step {step}")

Next steps
----------

- Wrap existing PyOD or scikit-learn fault classifiers for streaming via :doc:`pyod`.
- Add smoothing and conformal alert rules in :doc:`alerts`.
- Package stateful workers behind an HTTP endpoint in :doc:`serving`.
