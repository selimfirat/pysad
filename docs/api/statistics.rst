Statistics
==========

.. module:: pysad.statistics

This module holds running statistics. A running statistic changes as you add values with ``update``, and ``get`` returns its current value. Removable meters also support removing past values with ``remove`` (and can be windowed with :class:`RunningStatistic`); :class:`ADWINMeter`, :class:`CUSUMMeter`, :class:`DDMMeter`, :class:`EDDMMeter`, :class:`ExponentialMovingAverageMeter`, :class:`FHDDMMeter`, :class:`FHDDMSAddMeter`, :class:`FHDDMSMeter`, :class:`HDDMAMeter`, :class:`HDDMWMeter`, :class:`KSWINMeter`, :class:`OnlinePeriodicityMeter`, :class:`PageHinkleyMeter`, and :class:`QuantileMeter` manage their own state without supporting ``remove``.

:class:`CUSUMMeter` tracks shifts from a fixed target mean. Its ``get()`` method
returns the larger cumulative deviation. Its ``drift_detected`` flag reports
when this value exceeds ``threshold``. The next update restarts both sums.
Use :class:`PageHinkleyMeter` to track deviations from a running mean instead.
This module holds running statistics. Call ``update`` to add a value, then call
``get`` to read the statistic. Removable meters also support ``remove`` and can use
:class:`RunningStatistic` for windowed summaries. Each meter's class documentation
describes its supported operations.

:class:`DDSketchMeter` also supports ``merge(other)`` to combine summaries of
partitioned streams without replaying the observations. The meters must have the
same ``relative_accuracy``. The receiver retains its ``q`` and ``max_buckets``,
and the source stays unchanged. Counts remain exact; quantiles whose buckets have
been collapsed in either meter may fall outside the relative-error guarantee.
See ``examples/example_ddsketch_merge.py`` for a partitioned-stream example.

:class:`RelativeErrorQuantileMeter` tracks relative rank accuracy with randomized
compactions from :cite:`cormode2021relative`. This differs from DDSketch's relative
value accuracy. Set ``high_rank_accuracy=True`` to protect the upper tail.
Larger ``k`` improves accuracy. ``max_items`` bounds the stream count and memory.
Updates and merges reject counts above this limit. Merging requires equal ``k``,
``max_items``, and ``high_rank_accuracy``. The receiver keeps its ``q``, and the
source stays unchanged. The meter uses the paper's fixed stream limit variant.
It does not use the changing capacities for unknown stream lengths.

.. autosummary::
   :nosignatures:
   :template: class.rst
   :toctree: ../generated
   :tail: AbsStatistic, RunningStatistic
