Choose a detector
=================

No detector is the best for all data. Use the table to find two or three candidates, then compare them on your own data.

Find candidates
---------------

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
     - :class:`~pysad.models.MedianAbsoluteDeviation`, or :class:`~pysad.models.StandardAbsoluteDeviation` for a long stream
     - Outliers in the history have little effect on MAD, but it keeps every value, so its time per point grows with the stream. The standard deviation version takes constant time
   * - Several features, first try
     - :class:`~pysad.models.LODA`
     - Fast, has few parameters and learns from every point
   * - Features with known value ranges
     - :class:`~pysad.models.HalfSpaceTrees`
     - Uses the range of each feature. Constant time per point, scored against the previous window
   * - Many features
     - :class:`~pysad.models.xStream`
     - Projects the points to fewer features with hashing, so it handles many features
   * - Network traffic
     - :class:`~pysad.models.KitNet`
     - Designed for live network intrusion detection. With the defaults, it gives 0 for the first 50,000 points and stops learning after 100,000. Lower ``grace_feature_mapping`` and ``grace_anomaly_detector`` for a shorter stream
   * - A PyOD model that you already use
     - :class:`~pysad.models.integrations.ReferenceWindowModel`
     - Refits the model on a sliding window, see :doc:`pyod`

The table follows the design of each method. It is not a benchmark ranking. :doc:`/detectors` lists every detector with its input, its memory and its paper.

Compare the candidates
----------------------

Run each candidate on a labeled part of your data and compare a metric. This example compares four detectors on the ``cardio`` benchmark dataset:

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

              LODA: AUROC 0.951
           xStream: AUROC 0.598
    HalfSpaceTrees: AUROC 0.823
            RSHash: AUROC 0.700

The streamer keeps the order of the file, and in ``cardio`` all anomalies are at the end. ``fit_score_partial`` uses the evaluation order of each detector's paper: :class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.RSHash` score each point before they learn from it, and :class:`~pysad.models.LODA` and :class:`~pysad.models.xStream` learn first. The results on your data can be very different, so run the comparison on your own stream.

:class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.RSHash` need the range of each feature. :func:`~pysad.utils.get_minmax_array` computes it from data. In production, compute the range from historical data or use known physical limits.

Follow these rules to make the comparison fair:

- Use the same stream, in the same order, for every detector.
- Fix ``random_state``, and repeat the comparison with three to five seeds. Randomized detectors give different results for each seed.
- If you have no labels, add synthetic anomalies to a copy of your data. Or label some of the alerts by hand.

:doc:`evaluate` explains the metrics.

Tune the parameters
-------------------

Most detectors work with their default parameters. These parameters have the largest effect:

``window_size``
   The number of recent points that the detector uses. A larger window gives more stable scores. A smaller window follows changes in the data faster.

``num_components``, ``num_trees``, ``num_random_cuts``
   The size of an ensemble. A larger ensemble gives more stable scores and takes more time per point.

``period``
   For seasonal detectors, the number of points in one cycle. For example, use 24 for hourly data with a daily cycle.

Next steps
----------

- :doc:`evaluate` explains the metrics for the comparison.
- :doc:`alerts` turns the scores into alerts.
