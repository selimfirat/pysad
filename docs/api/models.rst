Models
======

.. module:: pysad.models

This module holds the streaming anomaly detectors. :doc:`/detectors` compares them. :doc:`/guides/choose_detector` helps you choose one.

``ADeMS`` implements AnomDetect with the deterministic DetUpdate sketch. It scores raw, uncentered reconstruction error before learning each instance. Use ``initial_X`` for trusted normal training data when filtering updates with ``threshold``; without a threshold, the first stream instances bootstrap the model. See ``examples/example_adems.py``.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated

   ADeMS
   AnyOut
   ApproxStorm
   ChangeFinder
   CPOD
   D3
   DAMP
   DILOF
   ExactStorm
   EXPoSE
   HalfSpaceTrees
   HoltWinters
   HTM
   IForestASD
   IncrementalLOF
   Inqmad
   KELOS
   KitNet
   KNNCAD
   LODA
   LocalOutlierProbability
   MatrixProfile
   MCOD
   MedianAbsoluteDeviation
   MemStream
   MIDAS
   MStream
   NETS
   OnlineIsolationForest
   OnlineOneClassSVM
   RelativeEntropy
   RobustRandomCutForest
   RSForest
   RSHash
   SAND
   SeasonalESD
   SeasonalHybridESD
   SmartSifter
   SpectralResidual
   StandardAbsoluteDeviation
   TEDA
   ThreshLEAP
   xStream
   NullModel
   PerfectModel
   RandomModel
   StreamPipeline
