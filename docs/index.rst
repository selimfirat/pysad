:hide-toc:

PySAD
=====

.. container:: hero

   .. image:: _static/logo.svg
      :alt: PySAD
      :width: 360px
      :class: only-light

   .. image:: _static/logo-dark.svg
      :alt: PySAD
      :width: 360px
      :class: only-dark

   .. rst-class:: tagline

      One pass. Every outlier.

   .. rst-class:: lead

      Streaming anomaly detection in Python. PySAD scores each data point when it arrives, with 16 streaming detectors, calibrated alerts and streaming evaluation.

   .. container:: actions

      :doc:`Get started <getting_started>` `View on GitHub <https://github.com/selimfirat/pysad>`_

.. container:: terminal

   .. code-block:: console

      $ pip install pysad

.. code-block:: python

   from pysad.evaluation import AUROCMetric
   from pysad.models import LODA
   from pysad.utils import Data

   model = LODA(random_state=0)
   metric = AUROCMetric()

   for x, label in Data().get_iterator("cardio.mat", shuffle=False):
       score = model.fit_score_partial(x)  # learn from x, then score it
       metric.update(label, score)

   print(f"AUROC: {metric.get():.3f}")

.. code-block:: text

   AUROC: 0.951

.. container:: stream-figure

   .. image:: images/stream-light.svg
      :alt: Three sensor readings stream in from left to right. A Robust Random Cut Forest scores each point when it arrives, and conformal calibration raises alerts at a spike, a level shift and a dip.
      :class: only-light

   .. image:: images/stream-dark.svg
      :alt: Three sensor readings stream in from left to right. A Robust Random Cut Forest scores each point when it arrives, and conformal calibration raises alerts at a spike, a level shift and a dip.
      :class: only-dark

   .. rst-class:: caption

      A Robust Random Cut Forest scores three synthetic sensors one point at a time, and conformal calibration raises an alert when the calibrated value reaches 0.995.

A batch detector needs the whole dataset before it can score a point. A stream does not give you the whole dataset. Points arrive one at a time, the data drifts, and you must decide before the next point arrives. Every PySAD detector takes one point at a time, and most keep a fixed amount of memory.

.. container:: cards

   .. container:: card

      :doc:`getting_started`

      Build your first streaming detector in ten minutes: score, measure and alert.

   .. container:: card

      :doc:`How-to guides <guides/index>`

      Recipes for real tasks: choose a detector, stream your data, run PyOD models, write your own detector.

   .. container:: card

      :doc:`concepts`

      How streaming detection works: the model interface, prequential evaluation, calibration and memory.

   .. container:: card

      :doc:`detectors`

      All 16 detectors, with the input they take, the memory they keep and the paper they implement.

   .. container:: card

      :doc:`api`

      Every public class, parameter and method.

   .. container:: card

      :doc:`examples`

      Complete scripts that run end to end. The automatic tests run every one of them.

What PySAD contains
-------------------

.. container:: stats

   .. container:: stat

      .. rst-class:: number

         16

      streaming detectors from published papers

   .. container:: stat

      .. rst-class:: number

         17

      labeled benchmark datasets, downloaded on first use

   .. container:: stat

      .. rst-class:: number

         60+

      publications that cite PySAD

- Detectors from robust statistics to random forests and random projections. :doc:`detectors` lists each one with its method, input and memory.
- Calibrators convert raw scores into the fraction of recent scores that they exceed, so one threshold works for every detector.
- Streamers replay arrays and DataFrames as streams. Metrics update after every point: :doc:`guides/evaluate` explains each one.
- Preprocessors, random projections, score postprocessors and ensemblers work with every detector.
- :class:`~pysad.models.integrations.ReferenceWindowModel` runs any `PyOD <https://github.com/yzhao062/pyod>`_ detector on a sliding window.

PySAD is free software under the BSD 3-Clause License. If you use it in research, see :doc:`citing`.

.. container:: mascot

   .. image:: images/tripod.svg
      :alt: Tripod, the PySAD mascot: a tripodfish that stands still on three long fins and waits for what the current brings.
      :width: 120px

   Tripod the tripodfish stands still on three fins and waits for what the current brings.

.. toctree::
   :hidden:
   :caption: Learn

   getting_started
   installation
   concepts
   examples

.. toctree::
   :hidden:
   :caption: How-to guides

   guides/index
   guides/stream_data
   guides/choose_detector
   guides/alerts
   guides/evaluate
   guides/pipelines
   guides/seasonal
   guides/pyod
   guides/production
   guides/custom_model

.. toctree::
   :hidden:
   :caption: Reference

   detectors
   api
   references

.. toctree::
   :hidden:
   :caption: Project

   faq
   contributing
   inclusion_criteria
   citing
   license
   GitHub <https://github.com/selimfirat/pysad>
   Changelog <https://github.com/selimfirat/pysad/releases>
