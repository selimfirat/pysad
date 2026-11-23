Concepts
========

This page explains streaming detection, pipeline components and evaluation. For instructions, see :doc:`getting_started` and the :doc:`how-to guides <guides/index>`.

Streaming anomaly detection
---------------------------

A stream is a sequence of records processed in arrival order, potentially indefinitely. An anomaly departs from the expected pattern: a sensor spike, fraudulent payment or failing server.

A batch detector, such as those in `PyOD <https://github.com/yzhao062/pyod>`_, trains on a fixed dataset and can then score new points. Updating that model generally requires retraining. An ongoing stream introduces three concerns:

- Points arrive one at a time, so scoring and learning must keep up with the arrival rate.
- Keeping every point makes memory grow without bound.
- Normal behavior can change over time (drift), making earlier data less representative.

A streaming detector updates its model of normal behavior with each point and assigns an anomaly score. Higher scores mean more anomalous points. It stores a summary, often a fixed-size window of recent points or scores.

The pipeline
------------

A PySAD pipeline is a loop over the stream. Each point goes through the same stages:

.. image:: images/pipeline-light.svg
   :alt: A point flows from the stream through optional preprocessing to a detector, which gives it an anomaly score. Optional postprocessing and ensembling smooth or combine the scores. A calibrator converts the score into a value between 0 and 1, and a threshold raises an alert. A metric measures the scores while the stream runs.
   :class: only-light pipeline-wide

.. image:: images/pipeline-dark.svg
   :alt: A point flows from the stream through optional preprocessing to a detector, which gives it an anomaly score. Optional postprocessing and ensembling smooth or combine the scores. A calibrator converts the score into a value between 0 and 1, and a threshold raises an alert. A metric measures the scores while the stream runs.
   :class: only-dark pipeline-wide

.. image:: images/pipeline-vertical-light.svg
   :alt: A point flows from the stream through optional preprocessing to a detector, which gives it an anomaly score. Optional postprocessing and ensembling smooth or combine the scores. A calibrator converts the score into a value between 0 and 1, and a threshold raises an alert. A metric measures the scores while the stream runs.
   :class: only-light pipeline-narrow

.. image:: images/pipeline-vertical-dark.svg
   :alt: A point flows from the stream through optional preprocessing to a detector, which gives it an anomaly score. Optional postprocessing and ensembling smooth or combine the scores. A calibrator converts the score into a value between 0 and 1, and a threshold raises an alert. A metric measures the scores while the stream runs.
   :class: only-dark pipeline-narrow

Only the stream and detector are required. Each stage manages its own state and can be added, removed or replaced independently.

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * - Stage
     - Module
     - What it does
   * - Stream
     - :mod:`pysad.utils`
     - Gives the points one at a time. Your own loop over a file, a socket or a queue also works.
   * - Preprocess
     - :mod:`pysad.transform.preprocessing`, :mod:`pysad.transform.projection`
     - Scales a point, removes seasonality, extracts features from raw inputs (for example with :class:`~pysad.transform.preprocessing.DrainLogParser` or :class:`~pysad.transform.preprocessing.FunctionTransformer`), or reduces the number of features.
   * - Detect
     - :mod:`pysad.models`
     - Learns from the point and gives it an anomaly score.
   * - Postprocess
     - :mod:`pysad.transform.postprocessing`
     - Smooths or normalizes the scores over a window.
   * - Ensemble
     - :mod:`pysad.transform.ensemble`
     - Combines the scores of several detectors into one score.
   * - Calibrate
     - :mod:`pysad.transform.probability_calibration`
     - Maps a score to a value between 0 and 1 using a reference distribution of scores.
   * - Evaluate
     - :mod:`pysad.evaluation`
     - Measures the scores against labels (anomaly or normal), one point at a time.

.. _model-interface:

The model interface
-------------------

Every detector extends :class:`~pysad.core.BaseModel` and has two single-point methods. The API reference calls a point an instance:

``fit_partial(x)``
   Updates the model with ``x``, a NumPy array of shape ``(num_features,)``. Retained points are copied, so callers can reuse the input array.

``score_partial(x)``
   Returns a Python ``float`` score without changing the model or modifying or retaining ``x``.

Most loops use ``fit_score_partial(x)``: score ``x`` using earlier points, then learn from it. This is prequential evaluation, or test-then-train. Before learning, detectors return a fixed score, usually 0. :class:`~pysad.models.SeasonalESD` and :class:`~pysad.models.SeasonalHybridESD` instead test the window containing the newest value, as their paper specifies.

:class:`~pysad.core.BaseModel` provides ``fit(X)``, ``score(X)`` and ``fit_score(X)`` for arrays of shape ``(num_instances, num_features)``. Rows are processed in order. Batch scores are NumPy ``float64`` arrays of shape ``(num_instances,)``.

Each model and postprocessor exposes ``output_kind`` (``"score"``, ``"probability"``, or ``"decision"``). Detectors with a paper-defined decision rule and pipelines ending in a decision postprocessor also provide ``predict_partial(x)``, ``fit_predict_partial(x)``, ``predict(X)``, and ``fit_predict(X)``, returning ``0`` or ``1`` (Python ``int`` for one instance, NumPy ``int64`` arrays for batches). See :doc:`guides/alerts`.

Preprocessors, projectors, postprocessors, ensemblers and calibrators follow the same pattern with ``fit_partial``, ``transform_partial`` and ``fit_transform_partial``. Running statistics, for example a running mean, use ``update`` and ``get``. Most of them also have ``remove``.

Calling only ``score_partial`` after training freezes the model while continuing to score new points.

Higher means more anomalous
---------------------------

Higher values mean more anomalous points across all PySAD components: detectors, postprocessors, score ensemblers, and probability calibrators. The alert rule stays the same when components change: raise an alert at or above the threshold.

Raw score scales differ between detectors (negative log densities, mass-based scores, displacement scores, or 0/1 decisions). Calibrate scores before comparing different detectors.

.. _score-convention:

Score direction convention
~~~~~~~~~~~~~~~~~~~~~~~~~~

PySAD uses a single, universal convention:

- **Higher score = more anomalous**.
- **Lower score = more normal**.
- All detectors return non-negative scores where larger values indicate greater anomalousness.
- All postprocessors, ensemblers, and calibrators preserve this ordering: if detector A scores an instance higher than detector B, the combined or calibrated score preserves that relationship.

Probability calibrators (``ConformalProbabilityCalibrator``, ``SPOTProbabilityCalibrator``, ``GaussianTailProbabilityCalibrator``, ``AnomalyLikelihoodProbabilityCalibrator``) return values in ``[0, 1]`` where higher values mean more anomalous. Internally they compute ``1 - p`` where ``p`` is the probability of being normal, so the output follows the same convention.

Model decisions and ``output_kind``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every model and postprocessor exposes an ``output_kind`` attribute with one of three values:

- ``"score"`` — continuous anomaly score (higher = more anomalous). This is the default for detectors.
- ``"probability"`` — calibrated anomaly probability in ``[0, 1]`` (higher = more anomalous). Returned by probability calibrators.
- ``"decision"`` — binary 0/1 output (1 = anomaly, 0 = normal). Returned by ``ThresholdFilter``, ``QuantileFilter``, ``FRAHST``, and ``RelativeEntropy``.

Detectors with a paper-defined decision threshold (``FluxEV``, ``GNGA``, ``OeSNNUAD``, ``NonparametricDynamicThresholdPostprocessor``) set ``_has_decision_rule = True`` and override ``_decide_from_score(score) -> int`` with their specific threshold logic. The default ``_decide_from_score`` for ``"decision"`` outputs returns ``1 if score > 0.0 else 0``.

Components with ``output_kind == "decision"`` or ``_has_decision_rule == True`` expose ``predict_partial``, ``fit_predict_partial``, ``predict``, and ``fit_predict`` returning ``int`` (0 or 1) or ``np.int64`` arrays. Calling ``predict*`` on a score-only model raises ``TypeError``. Piping a ``"decision"`` stage into a ``"probability"`` calibrator raises ``ValueError``.

For the five detectors documented in detail here:

- :class:`~pysad.models.HTM` returns the anomaly likelihood (0 to 1, higher is more anomalous) by default, or the raw anomaly score (fraction of unpredicted active columns) if ``likelihood=False``. A spatial anomaly (value outside the observed range by more than 5%) returns 1.0.
- :class:`~pysad.models.IDKS` returns the negated normal score (negative density), so higher scores mean more anomalous instances.
- :class:`~pysad.models.IForestASD` returns the isolation forest anomaly score minus the ``contamination`` quantile of the training scores of the current forest, so higher scores mean more anomalous instances.
- :class:`~pysad.models.Inqmad` returns the negative of the estimated density (quantum measurement), so higher scores (lower density) mean more anomalous instances.
- :class:`~pysad.models.KELOS` returns the negative KLOME z-score (how much lower the density at a point is than at its nearest micro-cluster centers), so higher scores mean more anomalous instances.

Calibration
-----------

A calibrator converts a raw score into a value between 0 and 1. PySAD has four calibrators:

:class:`~pysad.transform.probability_calibration.AnomalyLikelihoodProbabilityCalibrator`
   Gives the anomaly likelihood of HTM (Hierarchical Temporal Memory), the detector of Numenta. It fits a Gaussian to the averages of each 10 consecutive recent scores. It returns one minus the probability that such an average is further from the mean than the average of the last 10 scores, on the same side of the mean. The value is 0.5 at the mean and close to 1 far from it. It expects scores between 0 and 1.

:class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator`
   Gives the fraction of recent scores that are lower than the new score. A value of 0.99 means that the new score is higher than 99% of the scores in the window. In statistics, this value is one minus the conformal p-value. It makes no assumption about the distribution of the scores.

:class:`~pysad.transform.probability_calibration.GaussianTailProbabilityCalibrator`
   Fits a Gaussian distribution to recent scores using their mean and variance. Returns the probability of a lower score under this fit, assuming approximately Gaussian scores.

:class:`~pysad.transform.probability_calibration.SPOTProbabilityCalibrator`
   Fits a Generalized Pareto distribution to the highest scores with extreme value theory. It gives one minus the probability that a score is higher than the new score. It makes no assumption about the distribution of the scores, and it estimates the probability of scores that are higher than all scores seen so far.

A conformal threshold of 0.99 refers to rank within the recent window, not anomaly probability. About 1% of normal scores exceed it under a stable distribution with few ties. Drift, dependence, ties and anomalies in the window affect the observed rate. Raising the threshold reduces alerts, including false alerts on normal points.

Prequential evaluation
======================

Prequential evaluation measures performance on unseen points: score each point, record the score in a metric, then learn from the point.

.. code-block:: python

   for x, label in streamer.iter(X, y):
       score = model.score_partial(x)
       metric.update(label, score)
       model.fit_partial(x)

Most examples use ``fit_score_partial`` to combine scoring and learning. See :ref:`the model interface <model-interface>`. :func:`~pysad.evaluation.evaluate_stream` runs this loop on labeled data and returns each metric's final value.

The initial learning period is called warm-up. Full-stream metrics include it. To reveal changes over time, evaluate consecutive windows separately. See :doc:`guides/evaluate`.

Warm-up behavior of selected detectors
--------------------------------------

Each detector has a warm-up period before it produces meaningful scores. During warm-up, ``score_partial`` and ``fit_score_partial`` return a fixed value (usually 0.0 or 0.5) that does not reflect anomalousness.

- :class:`~pysad.models.HTM`: The anomaly likelihood is 0.5 for the first ``probationary_period`` points (default 750). The raw anomaly score (fraction of unpredicted columns) is available immediately if ``likelihood=False``. Spatial anomalies (values outside the observed range by more than 5%) return 1.0 at any time.
- :class:`~pysad.models.IDKS`: Returns 0.0 until the first window of ``window_size`` points (default 100) is full. After the window fills, the hypersphere partitions are initialized and scores reflect similarity to the window distribution.
- :class:`~pysad.models.IForestASD`: With ``drift_threshold=None`` (default), the forest is retrained on every completed window of ``window_size`` points (default 2048). Scores before the first retrain come from the initial forest (or the ``initial_window_X`` if provided). With a drift threshold, the first retrains happen at powers of two (1, 2, 4, 8, ...) until the first full window, then only when the window anomaly rate meets the threshold.
- :class:`~pysad.models.Inqmad`: The first ``initial_size`` points (default 256) form the initial sample. During this period, scores use the mean squared Gaussian kernel density of the sample so far (min-max scaled by the sample). Before any points are fitted, the score is 0.0 (the most anomalous possible score). After the initial sample completes, the adaptive Fourier features are fitted and the density matrix :math:`\rho` and threshold :math:`\tau` are initialized.
- :class:`~pysad.models.KELOS`: Returns 0.0 until the window has at least two micro-clusters with different densities. The window fills up to ``window_size`` points (default 1000). When all points in the window belong to a single cluster, or all k nearest centers have equal density, the z-score is undefined and the score is 0.0.

Memory and time
---------------

Most PySAD detectors keep a summary of the stream, not the stream itself. The summary is one of these:

- A window of recent points, for example in :class:`~pysad.models.ExactStorm`, whose window slides by one point, and :class:`~pysad.models.xStream`, which replaces its window every ``window_size`` points.
- Histograms or sketches (compact counts of where points fall), for example in :class:`~pysad.models.LODA` and :class:`~pysad.models.RSHash`.
- A set of trees, for example in :class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.RobustRandomCutForest`.
- Running statistics, for example in :class:`~pysad.models.StandardAbsoluteDeviation`.

These summaries bound memory and processing time independently of stream length. Some detectors retain growing state. :doc:`detectors` lists each detector's memory requirements.

Bounded-memory contracts for selected detectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following detectors have fixed or slowly growing memory that does not depend on stream length:

- :class:`~pysad.models.HTM`: Memory grows slowly. The temporal memory has at most 65,536 cells (``num_columns`` × ``num_cells_per_column``), each with up to 128 segments of 32 synapses. The anomaly likelihood calibrator keeps a window of 8,640 recent scores. With defaults (2048 columns, 32 cells), the upper bound is about 268 million synapses, but typical streams use far fewer.
- :class:`~pysad.models.IDKS`: Fixed memory. Keeps a sliding window of ``window_size`` points (default 100) and ``num_partitions × num_samples`` hypersphere samples (default 50 × 4 = 200 samples). Memory is independent of stream length.
- :class:`~pysad.models.IForestASD`: Fixed memory. Keeps two windows of ``window_size`` points (default 2048) — the reference window used to train the current forest, and the current window being filled. The isolation forest itself has ``n_estimators`` trees of at most ``max_samples`` nodes each.
- :class:`~pysad.models.Inqmad`: Fixed memory. Keeps a ``dim_x`` by ``dim_x`` density matrix (default 2000 × 2000 = 4 million entries, about 32 MB as float64), plus the random Fourier feature weights and offsets. The initial sample buffer of ``initial_size`` points (default 256) is released after initialization.
- :class:`~pysad.models.KELOS`: Fixed memory. Keeps a window of ``window_size`` points (default 1000) and the linear sums and counts of its micro-clusters (at most ``window_size`` clusters). Memory is independent of stream length.

Drift and forgetting
--------------------

Normal behavior changes over time. This change is drift. A detector follows drift only if it forgets old data. PySAD detectors handle old data in three ways:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - How it forgets
     - Detectors
     - Effect
   * - A window of recent data
     - :class:`~pysad.models.xStream`, :class:`~pysad.models.HalfSpaceTrees`, :class:`~pysad.models.ExactStorm`, :class:`~pysad.models.ApproxStorm`, :class:`~pysad.models.MCOD`, :class:`~pysad.models.NETS`, :class:`~pysad.models.CPOD`, :class:`~pysad.models.D3`, :class:`~pysad.models.ThreshLEAP`, :class:`~pysad.models.IncrementalLOF`, :class:`~pysad.models.KELOS`, :class:`~pysad.models.STARE`, :class:`~pysad.models.IForestASD`, :class:`~pysad.models.SeasonalESD`, :class:`~pysad.models.SeasonalHybridESD`, :class:`~pysad.models.KNNCAD`, :class:`~pysad.models.MatrixProfile`, :class:`~pysad.models.DAMP`, :class:`~pysad.models.MedianAbsoluteDeviation`, :class:`~pysad.models.SpectralResidual`
     - Data older than the window has no effect. A smaller window follows drift faster.
   * - Time decay
     - :class:`~pysad.models.RSHash`, :class:`~pysad.models.RobustRandomCutForest`, :class:`~pysad.models.Inqmad`, :class:`~pysad.models.OnlineOneClassSVM`, :class:`~pysad.models.ChangeFinder`, :class:`~pysad.models.AnyOut`, :class:`~pysad.models.HoltWinters`
     - Old points count less and less. A larger ``decay`` (``time_decay`` for :class:`~pysad.models.RobustRandomCutForest`, ``alpha`` for :class:`~pysad.models.Inqmad`, ``learning_rate`` for :class:`~pysad.models.OnlineOneClassSVM`, ``discount`` for :class:`~pysad.models.ChangeFinder`, ``alpha``, ``beta`` and ``gamma`` for :class:`~pysad.models.HoltWinters`) follows drift faster. :class:`~pysad.models.Inqmad` learns only from points that it classifies as normal. With the default ``time_decay=0``, :class:`~pysad.models.RobustRandomCutForest` keeps a uniform sample of the whole stream and does not forget.
   * - No forgetting
     - :class:`~pysad.models.LODA`, :class:`~pysad.models.ADeMS`, :class:`~pysad.models.StandardAbsoluteDeviation`, :class:`~pysad.models.RelativeEntropy`, :class:`~pysad.models.TEDA`
     - Every point since the start counts equally, so the detector reacts more slowly to drift as the stream grows.

:class:`~pysad.models.DILOF` keeps the newest half of its window. Each time the window is full, it replaces the oldest half with a summary half its size, so old points count less and less.

:class:`~pysad.models.KitNet` learns only during its grace periods, the first points of the stream. :class:`~pysad.models.LocalOutlierProbability` learns only from its initial points. After that, these two detectors do not follow drift.

A detector that does not forget can still follow a large, permanent change if you start a new detector after the change. :doc:`guides/production` shows how.

Reproducible results
--------------------

Randomized detectors, such as those using random projections or tree splits, accept ``random_state``. An ``int`` reproduces results. ``None`` (the default) uses NumPy’s global random state.

To make a complete script reproducible, pass ``random_state`` to each component, or call :func:`pysad.utils.fix_seed` once at the start. ``fix_seed`` seeds the global state that ``random_state=None`` uses.

Labels
------

Labels identify anomalous and normal points. Most detectors are unsupervised and ignore ``fit_partial``’s ``y`` parameter, which provides a common interface. Evaluation metrics use labels. Unsupervised detectors do not.
