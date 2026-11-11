Statistics
==========

.. module:: pysad.statistics

This module holds running statistics. A running statistic changes as you add values with ``update``, and ``get`` returns its current value. The meters also remove values with ``remove``, except :class:`ADWINMeter`, :class:`DDMMeter`, :class:`EDDMMeter`, :class:`FHDDMMeter`, :class:`HDDMAMeter`, :class:`HDDMWMeter`, :class:`KSWINMeter`, :class:`OnlinePeriodicityMeter` and :class:`PageHinkleyMeter`. :class:`RunningStatistic` removes old values itself, because it keeps only a window of the last values.

:class:`DDSketchMeter` also supports ``merge(other)`` to combine summaries of
partitioned streams without replaying the observations. The meters must have the
same ``relative_accuracy``. The receiver retains its ``q`` and ``max_buckets``,
and the source stays unchanged. Counts remain exact; quantiles whose buckets have
been collapsed in either meter may fall outside the relative-error guarantee.
See ``examples/example_ddsketch_merge.py`` for a partitioned-stream example.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated

   ADWINMeter
   AverageMeter
   CountMeter
   DDMMeter
   DDSketchMeter
   EDDMMeter
   ExponentialMovingAverageMeter
   FHDDMMeter
   HDDMAMeter
   HDDMWMeter
   KSWINMeter
   MaxMeter
   MedianMeter
   MinMeter
   OnlinePeriodicityMeter
   PageHinkleyMeter
   QuantileMeter
   SumMeter
   SumSquaresMeter
   VarianceMeter
   AbsStatistic
   RunningStatistic
