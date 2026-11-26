.. meta::
   :description: Compare the streaming drift detectors of PySAD: their papers, inputs, warning and drift flags, memory, and when to use each one.

PySAD drift detectors compared
==============================

A drift detector watches a stream and reports when the distribution of the points changes. PySAD keeps its drift detectors in :mod:`pysad.statistics`. Each meter takes one value with ``update`` and sets ``drift_detected`` after the update. Some meters also set ``warning_detected``. All meters use constant memory per update and do not store the whole stream.

This page compares the drift meters of PySAD side by side. It lists the paper of each meter, the input it expects, whether it reports warnings, its memory, and when to use it. The class docstring of each meter gives the full algorithm and the deviation from its paper.

Comparison
----------

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - Meter
     - Paper
     - Input
     - Warning
     - Drift
     - Two-sided
     - Memory
     - Best for
   * - :class:`~pysad.statistics.ADWINMeter`
     - Bifet and Gavaldà :cite:`bifet2007learning`
     - Values in [0, 1]
     - No
     - Yes
     - No
     - O(log W)
     - Unknown drift size
   * - :class:`~pysad.statistics.CUSUMMeter`
     - Page :cite:`page1954continuous`
     - Any numeric, fixed target mean
     - No
     - Yes
     - Yes
     - O(1)
     - Known reference mean
   * - :class:`~pysad.statistics.DDMMeter`
     - Gama et al. :cite:`gama2004learning`
     - 0/1 errors
     - Yes
     - Yes
     - No
     - O(1)
     - Classifier error stream
   * - :class:`~pysad.statistics.EDDMMeter`
     - Baena-García et al. :cite:`baenagarcia2006early`
     - 0/1 errors
     - Yes
     - Yes
     - No
     - O(1)
     - Classifier error stream
   * - :class:`~pysad.statistics.FHDDMMeter`
     - Pesaranghader and Viktor :cite:`pesaranghader2016fast`
     - 0/1 errors
     - No
     - Yes
     - No
     - O(window_size)
     - Abrupt drift
   * - :class:`~pysad.statistics.FHDDMSMeter`
     - Pesaranghader, Viktor, and Paquet :cite:`pesaranghader2018reservoir`
     - 0/1 errors
     - No
     - Yes
     - No
     - O(short + long)
     - Abrupt and gradual drift
   * - :class:`~pysad.statistics.FHDDMSAddMeter`
     - Pesaranghader, Viktor, and Paquet :cite:`pesaranghader2018reservoir`
     - 0/1 errors
     - No
     - Yes
     - No
     - O(stack_size)
     - Gradual drift, low memory
   * - :class:`~pysad.statistics.HDDMAMeter`
     - Frías-Blanco et al. :cite:`friasblanco2015online`
     - Values in [0, 1]
     - Yes
     - Yes
     - Yes, optional
     - O(1)
     - Mean shift in [0, 1]
   * - :class:`~pysad.statistics.HDDMWMeter`
     - Frías-Blanco et al. :cite:`friasblanco2015online`
     - Values in [0, 1]
     - Yes
     - Yes
     - Yes, optional
     - O(1)
     - Mean shift, recent values weigh more
   * - :class:`~pysad.statistics.KSWINMeter`
     - Raab et al. :cite:`raab2020reactive`
     - Any univariate numeric
     - No
     - Yes
     - No
     - O(window_size)
     - Spread or shape change
   * - :class:`~pysad.statistics.PageHinkleyMeter`
     - Page :cite:`page1954continuous`
     - Any numeric
     - No
     - Yes
     - Yes
     - O(1)
     - Gradual mean shift

Choose a meter
--------------

- If your stream is a sequence of 0/1 errors from a classifier, use :class:`~pysad.statistics.DDMMeter`, :class:`~pysad.statistics.EDDMMeter`, or :class:`~pysad.statistics.FHDDMMeter`.
- If your stream is numeric values in [0, 1], use :class:`~pysad.statistics.ADWINMeter`, :class:`~pysad.statistics.HDDMAMeter`, or :class:`~pysad.statistics.HDDMWMeter`.
- If you need to detect a change in the spread or the shape of the stream, not only the mean, use :class:`~pysad.statistics.KSWINMeter`.
- If you know the stable reference mean, use :class:`~pysad.statistics.CUSUMMeter`.
- If you track a running mean and want to detect a gradual shift, use :class:`~pysad.statistics.PageHinkleyMeter`.

All meters report drifts. Only :class:`~pysad.statistics.DDMMeter`, :class:`~pysad.statistics.EDDMMeter`, :class:`~pysad.statistics.HDDMAMeter`, and :class:`~pysad.statistics.HDDMWMeter` also report warnings. A warning comes before a drift and lets you prepare without resetting the model.

Assumptions
-----------

:class:`~pysad.statistics.ADWINMeter`, :class:`~pysad.statistics.HDDMAMeter`, and :class:`~pysad.statistics.HDDMWMeter` assume every value is in [0, 1]. Rescale values from a known range ``[a, b]`` to ``(x - a) / (b - a)`` before you pass them to ``update``. The meters do not check the range.

:class:`~pysad.statistics.KSWINMeter` is univariate. To watch several features, create one meter for each feature.

:class:`~pysad.statistics.CUSUMMeter` does not estimate a running mean. Set ``mean`` from a stable reference stream.

Example
-------

This script runs three meters on the same stream and prints the step where each one reports a drift:

.. code-block:: python

   from pysad.statistics import ADWINMeter, PageHinkleyMeter, KSWINMeter

   meters = [
       ADWINMeter(delta=0.002),
       PageHinkleyMeter(delta=0.0, threshold=50.0),
       KSWINMeter(alpha=0.0001, window_size=300, stat_size=30),
   ]

   for step, x in enumerate(stream):
       for meter in meters:
           meter.update(x)
           if meter.drift_detected:
               print(f"{meter.__class__.__name__}: drift at step {step}")

Next steps
----------

- :doc:`/guides/drift_aware_composition` resets a detector when a drift meter signals drift.
- :doc:`/guides/alerts` calibrates scores and raises threshold alerts.
- :doc:`/api/statistics` lists every running statistic.
