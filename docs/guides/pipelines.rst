Build a pipeline
================

A pipeline adds stages before and after the detector. Each stage is an object that learns from the stream, so you call it once per point, in the same loop as the detector. :doc:`/concepts` shows the stages in order.

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
   from pysad.transform.postprocessing import RunningAveragePostprocessor
   from pysad.transform.preprocessing import InstanceUnitNormScaler
   from pysad.transform.projection import GaussianRandomProjector

   scaler = InstanceUnitNormScaler()
   projector = GaussianRandomProjector(num_components=20, random_state=0)
   model = LODA(random_state=0)
   smoother = RunningAveragePostprocessor(window_size=5)

   for x in streamer.iter(X):
       x = scaler.fit_transform_partial(x)
       x = projector.fit_transform_partial(x)
       score = model.fit_score_partial(x)
       score = smoother.fit_transform_partial(score)

Preprocessors and projectors return a NumPy array, and postprocessors return a ``float``. Each ``fit_transform_partial`` call updates the stage with its input and then transforms the input.

If the points have many features, use a projection. A detector needs less time per point on fewer features, and random projections keep the distances between points approximately the same.

Combine several detectors
-------------------------

An ensemble combines the scores of several detectors into one score. It is usually more stable than each of its detectors.

The raw scores of different detectors have different scales. Normalize the scores of each detector first, for example with a running z-score, the distance from the recent mean in standard deviations:

.. code-block:: python

   import numpy as np

   from pysad.models import LODA, RSHash, xStream
   from pysad.transform.ensemble import AverageScoreEnsembler
   from pysad.transform.postprocessing import RunningZScorePostprocessor
   from pysad.utils import get_minmax_array

   mins, maxes = get_minmax_array(X)
   models = [
       LODA(random_state=0),
       xStream(random_state=0),
       RSHash(feature_mins=mins, feature_maxes=maxes, random_state=0),
   ]
   normalizers = [RunningZScorePostprocessor(window_size=500) for _ in models]
   ensembler = AverageScoreEnsembler()

   for x in streamer.iter(X):
       scores = np.array([
           normalizer.fit_transform_partial(model.fit_score_partial(x))
           for model, normalizer in zip(models, normalizers)
       ])
       score = ensembler.fit_transform_partial(scores)

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
     - The median score, which ignores a single outlying detector
   * - :class:`~pysad.transform.ensemble.AverageOfMaximumScoreEnsembler`
     - The average of the maximum score of random groups of detectors
   * - :class:`~pysad.transform.ensemble.MaximumOfAverageScoreEnsembler`
     - The maximum of the average score of random groups of detectors

For the last two ensemblers, the number of detectors must be a multiple of ``n_buckets`` (default 5), the number of groups.

Track running statistics
------------------------

The statistics in :mod:`pysad.statistics` track a value over a stream, for example to monitor the scores. Each meter has ``update``, ``remove`` and ``get``. :class:`~pysad.statistics.RunningStatistic` wraps a meter over a window and has ``update`` and ``get``:

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
