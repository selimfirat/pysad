Calibration
===========

.. module:: pysad.transform.probability_calibration

This module holds calibrators that convert scores into values between 0 and 1. A higher value means a more anomalous point. See :doc:`/guides/alerts`.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated

   ConformalProbabilityCalibrator
   GaussianTailProbabilityCalibrator
