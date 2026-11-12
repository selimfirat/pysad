Build a pipeline
================

A pipeline adds stages before and after the detector. Each stage is an object that learns from the stream. You call each stage once per point, in the same loop as the detector. :doc:`/concepts` shows the stages in order.

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

Preprocessors and projectors return a NumPy array, and postprocessors return a ``float``. Each ``fit_transform_partial`` call updates the stage with its input and then transforms the input. :class:`~pysad.transform.preprocessing.ShingleTransformer` is the exception, as the next section explains.

If the points have many features, project them to fewer features. A detector needs less time per point on fewer features. Random projections keep the distances between points approximately the same.

Give a detector the recent history
----------------------------------

A detector of single points sees only the values of each point. But many anomalies of a time series are a normal value at the wrong time, for example a high value at night. :class:`~pysad.transform.preprocessing.ShingleTransformer` turns each point into a shingle, the last ``shingle_size`` points side by side, so that the detector sees the recent pattern:

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

Choose ``shingle_size`` so that a shingle holds a typical pattern, for example one period of a seasonal series. ``fit_transform_partial`` of this stage builds the shingle first and then adds the point to its history, so a shingle holds only the point and the points before it.

Chain the stages into one model
-------------------------------

:class:`~pysad.models.StreamPipeline` runs the same loop in one model, and the ``|`` operator builds one directly. It has the methods of a detector, so it goes where a detector goes, for example into an evaluation loop or an ensemble:

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

Call ``score_one`` before ``learn_one`` for each point. The wrapper scores without fitting the PySAD model, and preserves its score range and warm-up behavior. River controls when the other pipeline stages learn.

The first dictionary fixes the feature order. Later dictionaries may list the keys in any order, but missing or new keys raise ``ValueError``. Pass ``feature_names`` to set the column order explicitly, for example ``feature_names=("cpu", "latency")``. The feature mapping stays fixed in size as the stream grows.

Pass a model class, such as ``RiverAnomalyDetector(LODA)``, or a zero-argument factory that creates a fresh model with custom parameters, as above. The adapter exposes the created model as ``model_``. River's ``clone()`` returns an untrained adapter, resetting inferred feature names and retaining explicit ones. Use ``clone(include_attributes=True)`` to copy both learned state and feature mapping, including in pipelines.

See ``examples/example_river_pipeline.py`` for a complete evaluation loop.

Combine several detectors
-------------------------

An ensemble combines the scores of several detectors into one score. It is usually more stable than each of its detectors.

The raw scores of different detectors have different scales. Before you combine the scores, normalize the scores of each detector with :class:`~pysad.transform.postprocessing.ZScorePostprocessor` over a sliding window, then combine the pipelines with ``&`` (or :class:`~pysad.models.ModelEnsemble` for a custom ensembler):

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

A meter tracks one statistic of a stream of values, for example the mean of the scores. The meters are in :mod:`pysad.statistics`. Each meter has ``update``, ``remove`` and ``get``. :class:`~pysad.statistics.RunningStatistic` applies a meter to a window of recent values and has ``update`` and ``get``:

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
