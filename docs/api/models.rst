Detectors
=========

.. module:: pysad.models

This module holds the streaming anomaly detectors. :doc:`/detectors` compares them, and :doc:`/guides/choose_detector` helps you choose one.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated

   ExactStorm
   HalfSpaceTrees
   IForestASD
   Inqmad
   KitNet
   KNNCAD
   LODA
   LocalOutlierProbability
   MedianAbsoluteDeviation
   RelativeEntropy
   RobustRandomCutForest
   RSHash
   SeasonalESD
   SeasonalHybridESD
   StandardAbsoluteDeviation
   xStream
   NullModel
   PerfectModel
   RandomModel
