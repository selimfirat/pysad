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

Preprocessors and projectors return NumPy arrays; postprocessors return ``float`` values. ``fit_transform_partial`` fits before transforming, except for :class:`~pysad.transform.preprocessing.ShingleTransformer`, described below.

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

Use a PySAD model in River
--------------------------------

:class:`~pysad.integrations.RiverAnomalyDetector` exposes a PySAD model through River's ``learn_one`` and ``score_one`` methods. It requires Python 3.11 or newer and works as the final stage of a River pipeline:

.. code-block:: python

   from functools import partial

   from river import preprocessing

   from pysad.integrations import RiverAnomalyDetector
   from pysad.models import LODA

   pipeline = preprocessing.StandardScaler() | RiverAnomalyDetector(partial(LODA, random_state=0))

   for x in streamer.iter(X):
       features = dict(enumerate(x))
       score = pipeline.score_one(features)
       pipeline.learn_one(features)

Call ``score_one`` before ``learn_one``. The wrapper preserves PySAD’s score range and warm-up behavior; River controls learning in the other stages.

The first dictionary fixes feature order and mapping size. Later keys may be reordered, but missing or new keys raise ``ValueError``. Set the order explicitly with ``feature_names``, for example ``feature_names=("cpu", "latency")``.

Pass a model class, such as ``RiverAnomalyDetector(LODA)``, or a zero-argument factory as above. Access the model as ``model_``. River’s ``clone()`` resets training and inferred feature names, retaining explicit names. ``clone(include_attributes=True)`` copies learned state and feature mapping, including in pipelines.

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
