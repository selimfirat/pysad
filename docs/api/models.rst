Models
======

.. module:: pysad.models

This module holds the streaming anomaly detectors. :doc:`/detectors` compares them. :doc:`/guides/choose_detector` helps you choose one.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated

   ExactStorm
   HalfSpaceTrees
   IForestASD
   IncrementalLOF
   Inqmad
   KitNet
   KNNCAD
   LODA
   LocalOutlierProbability
   MCOD
   MedianAbsoluteDeviation
   OnlineIsolationForest
   RelativeEntropy
   RobustRandomCutForest
   RSHash
   SeasonalESD
   SeasonalHybridESD
   SmartSifter
   StandardAbsoluteDeviation
   xStream
   NullModel
   PerfectModel
   RandomModel
   StreamPipeline
