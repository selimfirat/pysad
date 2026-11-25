.. meta::
   :description: Choose a running statistic in pysad.statistics, see which statistics support remove, bound memory with keep_values, and window a statistic with RunningStatistic.

Track running statistics
========================

Update a statistic one value at a time, then read its current value with ``get``. Every statistic in :mod:`pysad.statistics` follows this pattern.

Choose a statistic
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Statistic
     - Use it to
     - Memory
   * - :class:`~pysad.statistics.AverageMeter`
     - Track the mean of the values
     - O(1)
   * - :class:`~pysad.statistics.VarianceMeter`
     - Track the population variance with Welford's update
     - O(1)
   * - :class:`~pysad.statistics.MaxMeter` and :class:`~pysad.statistics.MinMeter`
     - Track the running maximum or minimum
     - O(1) with ``keep_values=False``
   * - :class:`~pysad.statistics.QuantileMeter`
     - Track an arbitrary quantile with the P^2 algorithm
     - O(1)
   * - :class:`~pysad.statistics.RunningStatistic`
     - Wrap a removable statistic to track it over a sliding window
     - O(window_size)

See :doc:`/api` for every statistic, parameter and method.

Removable and non-removable statistics
--------------------------------------

Every statistic implements ``update(num)`` and ``get()``. A removable statistic also implements ``remove(num)``, which deletes one earlier value from the statistic. Use removable statistics when a window slides over the stream, because the window removes the oldest value when a new one arrives.

:class:`~pysad.statistics.AverageMeter`, :class:`~pysad.statistics.VarianceMeter`, :class:`~pysad.statistics.MedianMeter` and :class:`~pysad.statistics.DDSketchMeter` are removable. :class:`~pysad.statistics.MaxMeter` and :class:`~pysad.statistics.MinMeter` are removable only with ``keep_values=True``. See the next section.

:class:`~pysad.statistics.QuantileMeter` is not removable. The P^2 algorithm keeps only five marker values, not the observations, so it cannot remove one. The same holds for the sketches, the drift meters and the exponential moving average meters. Wrapping a non-removable statistic raises ``TypeError``:

.. code-block:: python

   from pysad.statistics import QuantileMeter, RunningStatistic

   RunningStatistic(QuantileMeter, window_size=100)
   # TypeError: QuantileMeter does not support remove() and cannot be wrapped in RunningStatistic.

For a windowed quantile, wrap :class:`~pysad.statistics.MedianMeter` or :class:`~pysad.statistics.DDSketchMeter` instead.

Bound the memory with keep_values
----------------------------------

:class:`~pysad.statistics.MaxMeter` and :class:`~pysad.statistics.MinMeter` take a ``keep_values`` argument. With ``keep_values=False``, the meter keeps only the running extremum in O(1) memory, and ``remove`` raises ``ValueError``. With ``keep_values=True``, the meter stores every value in a deque, plus a deque of candidates, so memory grows with the number of values. Removal then works.

:class:`~pysad.statistics.RunningStatistic` passes ``keep_values=True`` automatically to any statistic that accepts it. In a window, memory stays bounded by ``window_size``, because each update removes the value that leaves the window.

Window a statistic with RunningStatistic
----------------------------------------

Wrap any removable statistic in :class:`~pysad.statistics.RunningStatistic` to track it over the most recent ``window_size`` values:

.. code-block:: python

   from pysad.statistics import AverageMeter, RunningStatistic

   stream = [1.0, 2.0, 3.0, 10.0, 4.0, 5.0, 6.0]
   windowed = RunningStatistic(AverageMeter, window_size=3)

   for value in stream:
       windowed.update(value)
       print(f"{value}: window average {windowed.get():.3f}")

.. code-block:: text

   1.0: window average 1.000
   2.0: window average 1.500
   3.0: window average 2.000
   10.0: window average 5.000
   4.0: window average 5.667
   5.0: window average 6.333
   6.0: window average 5.000

Each ``update`` appends the value to the window. If the window then holds more than ``window_size`` values, it removes the oldest one. ``get`` returns the statistic of the values still in the window.

Wrap any removable statistic the same way, for example ``RunningStatistic(MaxMeter, window_size=100)`` for the maximum of the last 100 values, or ``RunningStatistic(VarianceMeter, window_size=100)`` for their variance.

Next steps
----------

- :doc:`alerts` shows how to calibrate detector scores into alerts.
- :doc:`/api` lists every statistic with its parameters and methods.
