API reference
=============

This page lists every public class and function, grouped by module. Import each name from the module in its section, for example ``from pysad.models import LODA``.

Detectors
---------

.. currentmodule:: pysad.models

``pysad.models``: Streaming anomaly detectors. :doc:`detectors` compares them.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

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

PyOD integrations
-----------------

.. currentmodule:: pysad.models.integrations

``pysad.models.integrations``: Wrappers that run PyOD batch detectors on a stream.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   ReferenceWindowModel
   OneFitModel

Preprocessing
-------------

.. currentmodule:: pysad.transform.preprocessing

``pysad.transform.preprocessing``: Transformers that change each point before the detector gets it.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   IdentityScaler
   InstanceStandardScaler
   InstanceUnitNormScaler
   ModifiedSTLResidualTransformer
   SeasonalTrendDecomposer

Projection
----------

.. currentmodule:: pysad.transform.projection

``pysad.transform.projection``: Random projections that reduce the number of features.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   GaussianRandomProjector
   SparseRandomProjector
   StreamhashProjector

Postprocessing
--------------

.. currentmodule:: pysad.transform.postprocessing

``pysad.transform.postprocessing``: Postprocessors that smooth or normalize scores, over the full stream or over a window.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   AveragePostprocessor
   MaxPostprocessor
   MedianPostprocessor
   ZScorePostprocessor
   RunningAveragePostprocessor
   RunningMaxPostprocessor
   RunningMedianPostprocessor
   RunningZScorePostprocessor

Ensembling
----------

.. currentmodule:: pysad.transform.ensemble

``pysad.transform.ensemble``: Ensemblers that combine the scores of several detectors into one score.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   AverageScoreEnsembler
   MaximumScoreEnsembler
   MedianScoreEnsembler
   AverageOfMaximumScoreEnsembler
   MaximumOfAverageScoreEnsembler

Calibration
-----------

.. currentmodule:: pysad.transform.probability_calibration

``pysad.transform.probability_calibration``: Calibrators that convert scores into values between 0 and 1. Higher values mean more anomalous points.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   ConformalProbabilityCalibrator
   GaussianTailProbabilityCalibrator

Evaluation
----------

.. currentmodule:: pysad.evaluation

``pysad.evaluation``: Metrics that update one point at a time.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   AUROCMetric
   AUPRMetric
   PrecisionMetric
   RecallMetric
   WindowedMetric
   BaseSKLearnMetric

Statistics
----------

.. currentmodule:: pysad.statistics

``pysad.statistics``: Running statistics with ``update``, ``remove`` and ``get``.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   AverageMeter
   CountMeter
   MaxMeter
   MedianMeter
   MinMeter
   SumMeter
   SumSquaresMeter
   VarianceMeter
   AbsStatistic
   RunningStatistic

Utilities
---------

.. currentmodule:: pysad.utils

``pysad.utils``: Streamers, benchmark datasets and helpers.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   ArrayStreamer
   PandasStreamer
   Data
   Window
   fix_seed
   get_minmax_array
   get_minmax_scalar

Base classes
------------

.. currentmodule:: pysad.core

``pysad.core``: Abstract base classes. Subclass them to write your own components, as :doc:`guides/custom_model` shows.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   BaseModel
   BaseTransformer
   BasePostprocessor
   BaseMetric
   BaseStatistic
   BaseStreamer
