Choose a detector
=================

Choose two or three candidates from the table, then compare them on your data.

Find candidates
---------------

In the table, a window is the set of the most recent points that a detector keeps.

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Your data
     - Start with
     - Why
   * - One series of values with daily or weekly cycles
     - :class:`~pysad.models.SeasonalHybridESD`
     - Removes the cycle. Earlier anomalies in the window have little effect on the detector
   * - One series of values without a strong cycle
     - :class:`~pysad.models.MedianAbsoluteDeviation`
     - Anomalies in the window have little effect on the median absolute deviation (MAD). It uses only the last ``window_size`` values, so it follows drift
   * - Several features, first try
     - :class:`~pysad.models.LODA`
     - Is fast, has few parameters and learns from every point
   * - Features with known value ranges
     - :class:`~pysad.models.HalfSpaceTrees`
     - Uses the range of each feature. Takes constant time per point. Scores each point against the previous window
   * - Many features
     - :class:`~pysad.models.xStream`
     - Projects the points to fewer features with hashing, so it can use many features
   * - Network traffic
     - :class:`~pysad.models.KitNet`
     - Its authors made it for live detection of network intrusions. With the defaults, it gives 0 for the first 100 points and stops learning after 1,100. Set ``grace_feature_mapping`` and ``grace_anomaly_detector`` for the length of the stream
   * - A PyOD model that you already use
     - :class:`~pysad.integrations.ReferenceWindowModel`
     - Fits the model again on a sliding window. See :doc:`pyod`

These suggestions follow method design, not benchmark rank. See :doc:`/detectors` for inputs, memory and papers, and :doc:`benchmark </benchmark>` for multivariate results on |num_datasets| datasets.

Compare the candidates
----------------------

Compare candidates on labeled data using a metric such as AUROC, which measures ranking quality. This example compares four detectors on ``cardio``:

.. code-block:: python

   from pysad.evaluation import AUROCMetric
   from pysad.models import HalfSpaceTrees, LODA, RSHash, xStream
   from pysad.utils import ArrayStreamer, Data, get_minmax_array

   X, y = Data().get_data("cardio.mat")
   mins, maxes = get_minmax_array(X)

   candidates = {
       "LODA": LODA(random_state=0),
       "xStream": xStream(random_state=0),
       "HalfSpaceTrees": HalfSpaceTrees(feature_mins=mins, feature_maxes=maxes, random_state=0),
       "RSHash": RSHash(feature_mins=mins, feature_maxes=maxes, random_state=0),
   }

   for name, model in candidates.items():
       metric = AUROCMetric()
       for x, label in ArrayStreamer().iter(X, y):
           metric.update(label, model.fit_score_partial(x))
       print(f"{name:>15}: AUROC {metric.get():.3f}")

.. code-block:: text

              LODA: AUROC 0.946
           xStream: AUROC 0.859
    HalfSpaceTrees: AUROC 0.928
            RSHash: AUROC 0.700

This example preserves file order, with all ``cardio`` anomalies at the end, and scores before learning. Results depend on the data; repeat the comparison on your stream.

:class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.RSHash` require feature ranges. Compute them with :func:`~pysad.utils.get_minmax_array`; in production, use historical data or known physical limits.

Follow these rules to make the comparison fair:

- Use the same stream, in the same order, for every detector.
- Fix ``random_state``. Then repeat the comparison with three to five seeds (values of ``random_state``), because randomized detectors give different results for each seed.
- If you have no labels, add synthetic anomalies to a copy of your data. You can also label some of the alerts (points with a high score) by hand.

:doc:`evaluate` explains the metrics.

Tune the parameters
-------------------

Start with the defaults. If tuning is needed, focus on these parameters:

``window_size``
   The number of recent points that the detector uses. A larger window gives more stable scores. A smaller window follows changes in the data faster.

``num_trees``, ``num_random_cuts``, ``n_chains``, ``num_components``
   Ensemble size: larger ensembles generally stabilize scores but cost more time. RSHash uses ``num_components``. xStream uses ``n_chains`` for ensemble size and ``num_components`` for projected features.

``period``
   For seasonal detectors, the number of points in one cycle. For example, use 24 for hourly data with a daily cycle.

Next steps
----------

- :doc:`evaluate` explains the metrics for the comparison.
- :doc:`alerts` shows how to turn the scores into alerts.
