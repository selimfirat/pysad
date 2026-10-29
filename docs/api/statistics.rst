Statistics
==========

.. module:: pysad.statistics

This module holds running statistics. A running statistic changes as you add values with ``update``, and ``get`` returns its current value. The meters also remove values with ``remove``. :class:`RunningStatistic` removes old values itself, because it keeps only a window of the last values.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated

   AverageMeter
   CountMeter
   ExponentialMovingAverageMeter
   MaxMeter
   MedianMeter
   MinMeter
   PageHinkleyMeter
   QuantileMeter
   SumMeter
   SumSquaresMeter
   VarianceMeter
   AbsStatistic
   RunningStatistic
