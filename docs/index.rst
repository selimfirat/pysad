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

      PySAD is a Python library for anomaly detection on data streams. Score and learn from each point as it arrives, calibrate scores for alerts, and evaluate results against labels.

   .. container:: actions

      :doc:`Get started <getting_started>` `View on GitHub <https://github.com/selimfirat/pysad>`_ |stars|

.. |stars| image:: https://img.shields.io/github/stars/selimfirat/pysad?style=social
   :alt: GitHub stars
   :target: https://github.com/selimfirat/pysad

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
       score = model.fit_score_partial(x)  # score x, then learn from it
       metric.update(label, score)

   print(f"AUROC: {metric.get():.3f}")

.. code-block:: text

   AUROC: 0.946

.. container:: stream-figure

   .. image:: images/stream-light.svg
      :alt: Three sensor readings stream in from left to right. A Robust Random Cut Forest scores each point when the point arrives. Conformal calibration raises alerts at a spike, a short plateau and a dip.
      :class: only-light chart-wide

   .. image:: images/stream-dark.svg
      :alt: Three sensor readings stream in from left to right. A Robust Random Cut Forest scores each point when the point arrives. Conformal calibration raises alerts at a spike, a short plateau and a dip.
      :class: only-dark chart-wide

   .. image:: images/stream-narrow-light.svg
      :alt: Three sensor readings stream in from left to right. A Robust Random Cut Forest scores each point when the point arrives. Conformal calibration raises alerts at a spike, a short plateau and a dip.
      :class: only-light chart-narrow

   .. image:: images/stream-narrow-dark.svg
      :alt: Three sensor readings stream in from left to right. A Robust Random Cut Forest scores each point when the point arrives. Conformal calibration raises alerts at a spike, a short plateau and a dip.
      :class: only-dark chart-narrow

   .. rst-class:: caption

      A Robust Random Cut Forest scores three synthetic sensors one point at a time. Conformal calibration converts each score into the fraction of recent scores that it exceeds. It raises an alert when this calibrated value reaches 0.995.

Streaming detectors update as points arrive. They can process an ongoing stream without waiting for a complete dataset. Every PySAD detector takes one point at a time, and most keep a fixed amount of memory. See :doc:`concepts` for scoring order, drift and memory limits.

.. container:: cards

   .. container:: card

      :doc:`getting_started`

      Run your first streaming detector in ten minutes: score, measure and alert.

   .. container:: card

      :doc:`How-to guides <guides/index>`

      Steps for real tasks: choose a detector, stream your data, run PyOD models and write your own detector.

   .. container:: card

      :doc:`concepts`

      How streaming detection works: the detector interface, prequential evaluation (score each point, then learn from it), calibration and memory.

   .. container:: card

      :doc:`detectors`

      All detectors, with the input they take, the memory they keep and the paper they implement.

   .. container:: card

      :doc:`Benchmarks <benchmark>`

      AUROC, AUPR and time per point for all multivariate detectors across |num_datasets| labeled datasets.

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

         |num_detectors|

      streaming detectors from published papers

   .. container:: stat

      .. rst-class:: number

         |num_datasets|

      labeled :doc:`benchmark datasets <benchmark>`

   .. container:: stat

      .. rst-class:: number

         1

      alert threshold for every detector, after :doc:`conformal calibration <guides/alerts>`

   .. container:: stat

      .. rst-class:: number

         60+

      `publications that cite PySAD <https://scholar.google.com/scholar?cites=3884618599898298487>`_

- The detectors use robust statistics, random forests, random projections and other methods. :doc:`detectors` lists each detector with its method, input and memory, and the :doc:`benchmark <benchmark>` compares the multivariate detectors on |num_datasets| datasets.
- The conformal calibrator converts each raw score into the fraction of recent scores that it exceeds. The threshold therefore refers to a rank in the recent window rather than a detector-specific raw score. It does not specify the probability that a point is an anomaly.
- Streamers replay arrays and DataFrames as streams.
- Metrics update after every point. :doc:`guides/evaluate` explains each metric.
- Preprocessors, random projections, score postprocessors and ensemblers work with every detector. Preprocessors change each point before the detector scores it. Postprocessors change the scores, for example to smooth them. Ensemblers combine the scores of several detectors.
- :class:`~pysad.integrations.ReferenceWindowModel` runs any `PyOD <https://github.com/yzhao062/pyod>`_ detector on a sliding window (the most recent points).

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
   examples

.. toctree::
   :hidden:
   :caption: Explanation

   concepts

.. toctree::
   :hidden:
   :caption: How-to guides

   guides/index
   guides/stream_data
   guides/one_dimensional_integration
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
   :caption: Benchmarks

   benchmark
   benchmarks/setup

.. toctree::
   :hidden:
   :caption: Comparisons

   comparisons/index
   comparisons/river
   comparisons/capymoa
   comparisons/streamad
   comparisons/pyod
   comparisons/alibi_detect
   comparisons/darts

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
