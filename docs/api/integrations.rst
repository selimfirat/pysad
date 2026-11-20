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
   :exclude: PySADToRiverAdapter, RiverDriftDetector, PySADToScikitLearnAdapter
   :tail: ReferenceWindowModel, OneFitModel, PredictiveAnomalyDetector

River and scikit-learn pipeline adapters
----------------------------------------

.. autoclass:: PySADToRiverAdapter
   :exclude-members: clone, mutate

.. autoclass:: RiverDriftDetector
   :exclude-members: clone, mutate

.. autoclass:: PySADToScikitLearnAdapter
   :members: fit, score_samples, decision_function, predict
