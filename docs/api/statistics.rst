Statistics
==========

.. module:: pysad.statistics

This module holds running statistics. A running statistic changes as you add values with ``update``, and ``get`` returns its current value. Removable meters also support removing past values with ``remove`` (and can be windowed with :class:`RunningStatistic`); :class:`ADWINMeter`, :class:`DDMMeter`, :class:`EDDMMeter`, :class:`ExponentialMovingAverageMeter`, :class:`FHDDMMeter`, :class:`HDDMAMeter`, :class:`HDDMWMeter`, :class:`KLLMeter`, :class:`KSWINMeter`, :class:`OnlinePeriodicityMeter`, :class:`PageHinkleyMeter`, and :class:`QuantileMeter` manage their own state without supporting ``remove``.

:class:`DDSketchMeter` and :class:`KLLMeter` also support ``merge(other)`` to combine summaries of
partitioned streams without replaying the observations. For :class:`DDSketchMeter`, the meters must have the
same ``relative_accuracy``; the receiver retains its ``q`` and ``max_buckets``,
and the source stays unchanged. For :class:`KLLMeter`, compactors are merged and re-compacted,
bounding memory by :math:`O(k)`.
See ``examples/example_ddsketch_merge.py`` for a partitioned-stream example.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated
   :tail: AbsStatistic, RunningStatistic
