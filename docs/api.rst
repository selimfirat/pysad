API Reference
=============

This is the API documentation for ``PySAD``.


Core
^^^^

.. automodule:: pysad.core
    :no-members:
    :no-inherited-members:

.. currentmodule:: pysad

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

   core.BaseMetric
   core.BaseModel
   core.BasePostprocessor
   core.BaseStatistic
   core.BaseStreamer
   core.BaseTransformer


Individual Anomaly Models
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pysad.models
    :no-members:
    :no-inherited-members:

.. currentmodule:: pysad

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

    models.ExactStorm
    models.HalfSpaceTrees
    models.IForestASD
    models.Inqmad
    models.KitNet
    models.KNNCAD
    models.LODA
    models.LocalOutlierProbability
    models.MedianAbsoluteDeviation
    models.NullModel
    models.PerfectModel
    models.RandomModel
    models.RelativeEntropy
    models.RobustRandomCutForest
    models.RSHash
    models.StandardAbsoluteDeviation
    models.xStream


Integration Models
^^^^^^^^^^^^^^^^^^


.. automodule:: pysad.models.integrations
    :no-members:
    :no-inherited-members:

.. currentmodule:: pysad

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

    models.integrations.ReferenceWindowModel
    models.integrations.OneFitModel


Score Ensemblers
^^^^^^^^^^^^^^^^


.. automodule:: pysad.transform.ensemble
    :no-members:
    :no-inherited-members:

.. currentmodule:: pysad

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

    transform.ensemble.MaximumScoreEnsembler
    transform.ensemble.MedianScoreEnsembler
    transform.ensemble.AverageScoreEnsembler
    transform.ensemble.MaximumOfAverageScoreEnsembler
    transform.ensemble.AverageOfMaximumScoreEnsembler


Probability Calibrators
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pysad.transform.probability_calibration
    :no-members:
    :no-inherited-members:

.. currentmodule:: pysad

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

    transform.probability_calibration.ConformalProbabilityCalibrator
    transform.probability_calibration.GaussianTailProbabilityCalibrator


Projectors
^^^^^^^^^^

.. automodule:: pysad.transform.projection
    :no-members:
    :no-inherited-members:

.. currentmodule:: pysad

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

    transform.projection.StreamhashProjector
    transform.projection.GaussianRandomProjector
    transform.projection.SparseRandomProjector


Preprocessors
^^^^^^^^^^^^^

.. automodule:: pysad.transform.preprocessing
    :no-members:
    :no-inherited-members:

.. currentmodule:: pysad

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

    transform.preprocessing.IdentityScaler
    transform.preprocessing.InstanceStandardScaler
    transform.preprocessing.InstanceUnitNormScaler


Postprocessors
^^^^^^^^^^^^^^

.. automodule:: pysad.transform.postprocessing
    :no-members:
    :no-inherited-members:

.. currentmodule:: pysad

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

    transform.postprocessing.AveragePostprocessor
    transform.postprocessing.MaxPostprocessor
    transform.postprocessing.MedianPostprocessor
    transform.postprocessing.ZScorePostprocessor
    transform.postprocessing.RunningAveragePostprocessor
    transform.postprocessing.RunningMaxPostprocessor
    transform.postprocessing.RunningMedianPostprocessor
    transform.postprocessing.RunningZScorePostprocessor


Statistics
^^^^^^^^^^

.. automodule:: pysad.statistics
    :no-members:
    :no-inherited-members:

.. currentmodule:: pysad

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

    statistics.AbsStatistic
    statistics.RunningStatistic
    statistics.AverageMeter
    statistics.CountMeter
    statistics.MaxMeter
    statistics.MedianMeter
    statistics.MinMeter
    statistics.SumMeter
    statistics.SumSquaresMeter
    statistics.VarianceMeter


Evaluators
^^^^^^^^^^

.. automodule:: pysad.evaluation
    :no-members:
    :no-inherited-members:

.. currentmodule:: pysad

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

    evaluation.BaseSKLearnMetric
    evaluation.PrecisionMetric
    evaluation.RecallMetric
    evaluation.AUROCMetric
    evaluation.AUPRMetric
    evaluation.WindowedMetric


Utilities
^^^^^^^^^

.. automodule:: pysad.utils
    :no-members:
    :no-inherited-members:

.. currentmodule:: pysad

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: generated

    utils.Window
    utils.Data
    utils.ArrayStreamer
    utils.PandasStreamer
    utils._iterate
    utils.get_minmax_array
    utils.get_minmax_scalar
    utils.fix_seed

