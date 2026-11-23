Build a pipeline
================

A pipeline applies preprocessing, detection and postprocessing to each point. :doc:`/concepts` explains the stages.

The examples on this page use the ``arrhythmia`` benchmark dataset, which has 274 features:

.. code-block:: python

   from pysad.utils import ArrayStreamer, Data

   X, y = Data().get_data("arrhythmia.mat")
   streamer = ArrayStreamer()

Preprocess, detect, postprocess
-------------------------------

This pipeline scales each point to length 1, projects it to 20 features, scores it and smooths the score:

.. code-block:: python

   from pysad.models import LODA
   from pysad.transform.postprocessing import AveragePostprocessor
   from pysad.transform.preprocessing import InstanceUnitNormScaler
   from pysad.transform.projection import GaussianRandomProjector

   scaler = InstanceUnitNormScaler()
   projector = GaussianRandomProjector(num_components=20, random_state=0)
   model = LODA(random_state=0)
   smoother = AveragePostprocessor(window_size=5)

   for x in streamer.iter(X):
       x = scaler.fit_transform_partial(x)
       x = projector.fit_transform_partial(x)
       score = model.fit_score_partial(x)
       score = smoother.fit_transform_partial(score)

Preprocessors and projectors return NumPy arrays. Postprocessors return ``float`` values. ``fit_transform_partial`` fits before transforming, except for :class:`~pysad.transform.preprocessing.ShingleTransformer`, described below.

Projecting to fewer features reduces detection cost. Random projections approximately preserve distances.

Give a detector the recent history
----------------------------------

Some anomalies depend on context, such as a high value at night. :class:`~pysad.transform.preprocessing.ShingleTransformer` concatenates the latest ``shingle_size`` points into one vector so the detector sees recent patterns:

.. code-block:: python

   import numpy as np

   from pysad.models import LODA
   from pysad.transform.preprocessing import ShingleTransformer

   x = np.sin(np.arange(1000) * 2 * np.pi / 24)  # a daily cycle of 24 points
   shingler = ShingleTransformer(shingle_size=24)
   model = LODA(random_state=0)

   for value in x:
       shingle = shingler.fit_transform_partial(np.array([value]))  # the last 24 values
       score = model.fit_score_partial(shingle)

Set ``shingle_size`` to cover a typical pattern, such as one seasonal period. This stage’s ``fit_transform_partial`` builds the shingle before updating its history, using the current and earlier points.

Chain the stages into one model
-------------------------------

The ``|`` operator builds a :class:`~pysad.models.StreamPipeline` with the detector interface. Use it in evaluation loops or ensembles:

.. code-block:: python

   pipeline = (
       InstanceUnitNormScaler()
       | GaussianRandomProjector(num_components=20, random_state=0)
       | LODA(random_state=0)
       | AveragePostprocessor(window_size=5)
   )

   for x in streamer.iter(X):
       score = pipeline.fit_score_partial(x)

``fit_score_partial`` gives the same scores as the loop in the first section. ``score_partial`` scores a point with the current state of each stage and does not fit any stage. A calibrator is a postprocessor, so put it last in ``postprocessors``.

Wrap custom extractors or encoders
-----------------------------------

:class:`~pysad.transform.preprocessing.FunctionTransformer` turns a Python callable into a :class:`~pysad.core.BaseTransformer`, so custom feature extractors, domain parsers, or frozen embedding functions fit inside a ``|`` pipeline:

.. code-block:: python

   import numpy as np

   from pysad.models import LODA
   from pysad.transform.postprocessing import ThresholdFilter
   from pysad.transform.preprocessing import FunctionTransformer
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator


   def extract_features(record):
       return np.array([len(record), sum(c.isdigit() for c in record)], dtype=np.float64)


   pipeline = (
       FunctionTransformer(extract_features, output_dims=2)
       | LODA(random_state=0)
       | ConformalProbabilityCalibrator(window_size=200)
       | ThresholdFilter(threshold=0.95)
   )

   for record in ["ok 1", "ok 2", "error 99999999"]:
       alert = pipeline.fit_predict_partial(record)

Use frozen time-series embeddings
---------------------------------

A time-series foundation model (TSFM) can encode a context window as a fixed-length vector.
Use an existing pretrained encoder as a callable in :class:`~pysad.transform.preprocessing.FunctionTransformer`.
Leave ``fit_func=None`` so PySAD does not train the encoder.
The detector learns from the vectors through the same ``|`` interface.

This runnable recipe uses NumPy summary features as a stand-in for the encoder.
The stand-in is not a TSFM and does not measure TSFM detection quality.
It needs no model download or new dependency:

.. code-block:: python

   import numpy as np

   from pysad.models import LODA
   from pysad.transform.preprocessing import FunctionTransformer, ShingleTransformer
   from pysad.utils import ArrayStreamer


   def encode_window(context):
       """Return summary features as a stand-in for frozen embeddings.

       Args:
           context (np.ndarray): The context window, oldest value first.

       Returns:
           np.ndarray: A vector with the window mean and standard deviation.
       """
       return np.array([context.mean(), context.std()], dtype=np.float64)


   context_size = 24
   embedding_dims = 2
   pipeline = (
       ShingleTransformer(shingle_size=context_size)
       | FunctionTransformer(encode_window, fit_func=None, output_dims=embedding_dims)
       | LODA(random_state=0)
   )

   values = np.sin(np.arange(120) * 2 * np.pi / 24).reshape(-1, 1)
   scores = []
   for point in ArrayStreamer().iter(values):
       scores.append(pipeline.fit_score_partial(point))

Each input has shape ``(1,)``.
The shingler gives ``encode_window`` a vector of shape ``(context_size,)``, oldest value first.
It uses only the current point and earlier points.
During warm-up, it pads the window with copies of the oldest point.
The detector scores each vector before learning from it.
Higher scores mean more anomalous vectors.

To use a TSFM, replace ``encode_window`` with a callable around your existing pretrained encoder.
Load its weights once before the loop.
Use its inference mode, disable gradient updates, and keep its weights fixed.
``fit_func=None`` prevents PySAD fit callbacks but does not control the encoder's internal behavior.
Inside the callable, reshape the context to the encoder's expected input shape.
Apply any required normalization using only that context or fixed training statistics.
Return one numeric vector of shape ``(embedding_dims,)`` with the same width on every call.
Convert framework tensors to a CPU NumPy array before returning.
Set ``embedding_dims`` to the encoder's output width.
Keep ``context_size`` fixed to bound the history and encoder input size.
The encoder and its runtime stay outside PySAD's dependencies.

Use a PySAD model in River
--------------------------------

:class:`~pysad.integrations.PySADToRiverAdapter` exposes a PySAD model through River's ``learn_one`` and ``score_one`` methods. It works as the final stage of a River pipeline:

.. code-block:: python

   from functools import partial

   from river import preprocessing

   from pysad.integrations import PySADToRiverAdapter
   from pysad.models import LODA

   pipeline = preprocessing.StandardScaler() | PySADToRiverAdapter(partial(LODA, random_state=0))

   for x in streamer.iter(X):
       features = dict(enumerate(x))
       score = pipeline.score_one(features)
       pipeline.learn_one(features)

Call ``score_one`` before ``learn_one``. The adapter preserves PySAD’s score range and warm-up behavior. River controls learning in the other stages.

The first dictionary fixes feature order and mapping size. Later dictionaries can list the keys in any order, but missing or new keys raise ``ValueError``. Set the order explicitly with ``feature_names``, for example ``feature_names=("cpu", "latency")``.

Pass a model class, such as ``PySADToRiverAdapter(LODA)``, or a zero-argument factory as above. Access the model as ``model_``. River’s ``clone()`` resets training and inferred feature names, retaining explicit names. ``clone(include_attributes=True)`` copies learned state and feature mapping, including in pipelines.

See ``examples/example_river_pipeline.py`` for a complete evaluation loop.

Combine several detectors
-------------------------

An ensemble combines the scores of several detectors into one score. Compare its performance with the individual detectors on your data, as :doc:`evaluate` shows.

Normalize each detector’s scores with a windowed :class:`~pysad.transform.postprocessing.ZScorePostprocessor`, then combine pipelines with ``&``. Use :class:`~pysad.models.ModelEnsemble` for a custom ensembler:

.. code-block:: python

   from pysad.models import LODA, ModelEnsemble, RSHash, xStream
   from pysad.transform.ensemble import MaximumScoreEnsembler
   from pysad.transform.postprocessing import ZScorePostprocessor
   from pysad.utils import get_minmax_array

   mins, maxes = get_minmax_array(X)
   ensemble = (
       (LODA(random_state=0) | ZScorePostprocessor(window_size=500))
       & (xStream(random_state=0) | ZScorePostprocessor(window_size=500))
       & (
           RSHash(feature_mins=mins, feature_maxes=maxes, random_state=0)
           | ZScorePostprocessor(window_size=500)
       )
   )

   for x in streamer.iter(X):
       score = ensemble.fit_score_partial(x)

   max_ensemble = ModelEnsemble(ensemble.models, ensembler=MaximumScoreEnsembler())

PySAD has five ensemblers in :mod:`pysad.transform.ensemble`:

.. list-table::
   :header-rows: 1

   * - Ensembler
     - Score
   * - :class:`~pysad.transform.ensemble.AverageScoreEnsembler`
     - The average, or a weighted average, of the scores
   * - :class:`~pysad.transform.ensemble.MaximumScoreEnsembler`
     - The highest score, which is high if any detector gives a high score
   * - :class:`~pysad.transform.ensemble.MedianScoreEnsembler`
     - The median score, which ignores one detector with an extreme score
   * - :class:`~pysad.transform.ensemble.AverageOfMaximumScoreEnsembler`
     - The average of the maximum score of random groups of detectors
   * - :class:`~pysad.transform.ensemble.MaximumOfAverageScoreEnsembler`
     - The maximum of the average score of random groups of detectors

For the last two ensemblers, the number of detectors must be a multiple of ``n_buckets`` (default 5), the number of groups.

Track running statistics
------------------------

Meters in :mod:`pysad.statistics` track statistics with ``update``, ``remove`` and ``get``. :class:`~pysad.statistics.RunningStatistic` applies a meter to a recent window using ``update`` and ``get``:

.. code-block:: python

   from pysad.models import LODA
   from pysad.statistics import AverageMeter, RunningStatistic, VarianceMeter

   model = LODA()
   mean = AverageMeter()  # over the full stream
   recent_variance = RunningStatistic(VarianceMeter, window_size=100)  # over the last 100 values

   for x in streamer.iter(X):
       score = model.fit_score_partial(x)
       mean.update(score)
       recent_variance.update(score)

   print(f"mean: {mean.get():.2f}, recent variance: {recent_variance.get():.2f}")

Next steps
----------

- :doc:`alerts` calibrates the output of a pipeline.
- :doc:`evaluate` measures whether a stage improves the result.
