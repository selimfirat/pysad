Scikit-learn adapters
==============================

.. currentmodule:: pysad.integrations

Wrap a fitted PySAD model for batch scoring in a scikit-learn pipeline.
This adapter uses the existing scikit-learn dependency.

.. autoclass:: SklearnAnomalyDetector
   :members: fit, score_samples, decision_function, predict
