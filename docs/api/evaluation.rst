Evaluation
==========

.. module:: pysad.evaluation

This module holds streaming metrics. Each metric updates its value one point at a time. See :doc:`/guides/evaluate`.

:class:`BrierScoreMetric` measures mean squared error for binary anomaly probabilities.
Use label 1 for anomalies and 0 for normal instances. Lower scores mean better probabilities.
An empty stream returns ``0.0``. Invalid labels or probabilities raise ``ValueError``.

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
