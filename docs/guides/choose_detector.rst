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
   * - One metric with daily or weekly cycles
     - :class:`~pysad.models.SeasonalHybridESD`
     - Removes the seasonal pattern, and earlier anomalies in the window have little effect on it
   * - One metric without a strong cycle
     - :class:`~pysad.models.MedianAbsoluteDeviation`
     - A robust baseline that works with its default parameters
   * - Several features, first try
     - :class:`~pysad.models.LODA`
     - Fast, has few parameters and learns from every point
   * - Features with known value ranges
     - :class:`~pysad.models.HalfSpaceTrees`
     - Constant time per point over a sliding window
   * - Many features, or features that appear and disappear
     - :class:`~pysad.models.xStream`
     - Designed for high-dimensional streams whose features change
   * - Network traffic
     - :class:`~pysad.models.KitNet`
     - Designed for online network intrusion detection
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

:class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.RSHash` need the range of each feature. :func:`~pysad.utils.get_minmax_array` computes it from data. In production, compute the range from historical data or use known physical limits.

Follow these rules to make the comparison fair:

- Use the same stream, in the same order, for every detector.
- Fix ``random_state``, and repeat the comparison with three to five seeds. Randomized detectors give different results for each seed.
- If you have no labels, inject synthetic anomalies into a copy of your data, or label a sample of the alerts by hand.

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
