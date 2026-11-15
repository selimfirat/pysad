Evaluation
==========

.. module:: pysad.evaluation

This module holds streaming metrics. Each metric updates its value one point at a time. See :doc:`/guides/evaluate`.

:class:`BrierScoreMetric` measures mean squared error for binary anomaly probabilities.
Use label 1 for anomalies and 0 for normal instances. Lower scores mean better probabilities.
An empty stream returns ``0.0``. Invalid labels or probabilities raise ``ValueError``.

:class:`ExpectedCalibrationErrorMetric` measures calibration for binary anomaly probabilities.
It weights each bin's absolute difference between mean probability and anomaly frequency by its fraction of the stream.
Use label 1 for anomalies and 0 for normal instances. Lower scores mean better calibration.
The ``num_bins`` parameter sets fixed, equal-width bins (default 10).
Bin k covers ``[k / num_bins, (k + 1) / num_bins)``. The final bin includes ``1.0``.
Empty bins contribute zero. An empty stream returns ``0.0``.
Invalid labels or probabilities raise ``ValueError`` and leave the metric unchanged.
Counts, label sums, and probability sums use fixed storage that does not grow with the stream.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated
   :kind: classes
   :tail: BaseSKLearnMetric

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :kind: functions
