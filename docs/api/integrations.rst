Integrations
============

.. module:: pysad.integrations

This module holds bidirectional integrations between PySAD, PyOD, scikit-learn, and River: wrappers that run batch detectors and River forecasters on a stream, and adapters that expose PySAD detectors inside River and scikit-learn pipelines. See :doc:`/guides/pyod` and :doc:`/guides/pipelines`.

Stream wrappers for external models
-----------------------------------

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated
   :exclude: RiverAnomalyDetector, SklearnAnomalyDetector
   :tail: ReferenceWindowModel, OneFitModel, PredictiveAnomalyDetector

River and scikit-learn pipeline adapters
----------------------------------------

River adapters require Python 3.11 or newer.

.. autoclass:: RiverAnomalyDetector
   :exclude-members: clone, mutate

.. autoclass:: SklearnAnomalyDetector
   :members: fit, score_samples, decision_function, predict
