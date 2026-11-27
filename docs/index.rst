.. meta::
   :description: PySAD is a Python library for streaming anomaly detection on multivariate and univariate data streams, with 50+ online detectors, conformal and SPOT calibration, drift monitoring, prequential evaluation, and an MCP server with auto-discovered tools.

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

   .. container:: actions

      .. image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fselimfirat%2Fpysad%2Fmaster%2Fdocs%2F_static%2Fthroughput.json&style=flat-square&labelColor=0d1b22
         :target: benchmark.html
         :alt: Median throughput

      .. image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fselimfirat%2Fpysad%2Fmaster%2Fdocs%2F_static%2Fmemory.json&style=flat-square&labelColor=0d1b22
         :target: benchmarks/multivariate.html#memory-stability
         :alt: Maximum traced memory growth in three measured detectors

      .. image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fselimfirat%2Fpysad%2Fmaster%2Fdocs%2F_static%2Funivariate_detectors.json&style=flat-square&labelColor=0d1b22
         :target: detectors.html
         :alt: Univariate detectors

      .. image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fselimfirat%2Fpysad%2Fmaster%2Fdocs%2F_static%2Fmultivariate_detectors.json&style=flat-square&labelColor=0d1b22
         :target: detectors.html
         :alt: Multivariate detectors

   Throughput: median across recorded ODDS runs.
   Memory: maximum traced growth in three detectors over 1M points after warmup.

   .. container:: actions

      .. image:: https://img.shields.io/badge/Hugging%20Face-Open%20in%20Spaces-00757a?style=flat-square&logo=huggingface&logoColor=white&labelColor=0d1b22
         :target: https://huggingface.co/spaces/selimfirat/pysad-playground
         :alt: Open in Hugging Face Spaces

      .. image:: https://img.shields.io/badge/JupyterLite-Live%20Demo-00757a?style=flat-square&logo=jupyter&logoColor=white&labelColor=0d1b22
         :target: https://selimfirat.github.io/pysad/lab/index.html?path=nyc_taxi_alerts.ipynb
         :alt: Open in JupyterLite

      .. image:: https://img.shields.io/badge/Colab-Open%20Notebook-00757a?style=flat-square&logo=googlecolab&logoColor=white&labelColor=0d1b22
         :target: https://colab.research.google.com/github/selimfirat/pysad/blob/master/examples/quickstart.ipynb
         :alt: Open in Colab

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

Live dashboard, MCP, and optional kernels
-----------------------------------------

If you use uv, run the live dashboard with ``uvx``:

.. code-block:: console

   $ uvx --from pysad pysad-live --source synthetic

- The :doc:`live dashboard <api/live>` displays streaming scores and alerts in your terminal. Add ``--web`` to open the web dashboard, or use ``--source data.csv`` for your own data.
- :doc:`pysad.mcp <guides/mcp>` exposes detectors, drift meters, and explanations through the Model Context Protocol (MCP). Install ``pysad[mcp]``, then run ``pysad-mcp``. The server auto-discovers detectors from ``pysad.models.__all__`` and statistics from ``pysad.statistics.__all__``. Tool names use ``pysad_<lowercase_name>``. Each tool includes metadata for constructor parameters and defaults.

Optional PyO3 kernels (pysad_kernels)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the optional PyO3 Rust extension separately from PySAD:

.. code-block:: console

   $ pip install pysad_kernels

The extension provides kernels for four detectors:

* :class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.RobustRandomCutForest` use the available tree kernels automatically.
* :class:`~pysad.models.LODA` uses NumPy by default. Set ``use_rust=True`` to use the density kernel.
* :class:`~pysad.models.StreamKMPlusPlus` uses NumPy by default. Set ``use_rust=True`` to use the weighted Lloyd kernel. Sampling stays in Python. Floating-point rounding can change cluster assignments.

If a kernel is unavailable, the detector uses NumPy. Pickles store the option and model arrays, so they also work without the extension.

See :doc:`benchmarks/kernels` for the traversal timings of the optional kernels.

To build from a source checkout, install ``maturin`` and run these commands from the repository root. This requires a Rust toolchain.

.. code-block:: console

   $ pip install maturin
   $ maturin build --manifest-path rust/Cargo.toml --release --locked
   $ pip install rust/target/wheels/pysad_kernels-0.1.0-*.whl

Make sure that Python can import the extension:

.. code-block:: console

   $ python -c "import pysad_kernels; print(pysad_kernels.__file__)"

Migration guides
----------------

Use these guides to map your existing workflow to PySAD:

- :doc:`Azure AI Anomaly Detector and Metrics Advisor <guides/migrate_azure>`
- :doc:`AWS Lookout for Equipment and Kinesis RCF <guides/migrate_aws>`
- :doc:`Darts <guides/migrate_darts>`
- :doc:`Salesforce Merlion <guides/migrate_merlion>`
- :doc:`StatsForecast and Nixtla <guides/migrate_statsforecast>`
- :doc:`Alibi Detect <guides/migrate_alibi_detect>`
- :doc:`ADTK <guides/migrate_adtk>`
- :doc:`PyOD <guides/migrate_pyod>`
- :doc:`River <guides/migrate_river>`

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

      Mean AUROC and time per point for all multivariate detectors across |num_datasets| labeled datasets.

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
- :class:`~pysad.integrations.PySADToCapyMoaAdapter` exposes any PySAD detector through CapyMoa's streaming interface (`partial_fit`, `score`, `predict`, `predict_proba`, `reset`) without importing CapyMoa or needing Java.
- :doc:`benchmarks/run` shows how to compare PySAD detectors with River's streaming detectors through :class:`~pysad.integrations.PySADToRiverAdapter` using ``benchmarks/compare_river_adapter.py``.

PySAD is free software under the BSD 3-Clause License. If you use it in research, see :ref:`citing-pysad`.

.. container:: mascot

   .. image:: images/tripod.svg
      :alt: Tripod, the PySAD mascot: a tripodfish that stands still on three long fins and waits for what the current brings.
      :width: 120px

   Tripod the tripodfish stands still on three fins and waits for what the current brings.

.. toctree::
   :hidden:
   :caption: Learn

   getting_started
   cheatsheet
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
   guides/dataset_inventory
   guides/choose_detector
   guides/alerts
   guides/conformal_tsfm
   guides/evaluate
   guides/statistics
   guides/troubleshooting
   guides/pipelines
   guides/river_interoperability
   guides/drift_aware_composition
   guides/seasonal
   guides/pyod
   guides/production
   guides/serving
   guides/live_dashboard
   guides/custom_model
   guides/configuration
   guides/mcp
   guides/hardening
   guides/mcp_monitoring
   guides/migrate_azure
   guides/migrate_aws
   guides/migrate_darts
   guides/migrate_merlion
   guides/migrate_statsforecast
   guides/migrate_alibi_detect
   guides/migrate_adtk
   guides/migrate_pyod
   guides/migrate_river

.. toctree::
   :hidden:
   :caption: Benchmarks

   benchmark

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
   comparisons/adtk
   comparisons/statsforecast
   comparisons/drift

.. toctree::
   :hidden:
   :caption: Reference

   detectors
   detector_catalog
   api
   references

.. toctree::
   :hidden:
   :caption: Project

   faq
   contributing
   inclusion_criteria
   license
   pyopensci_review
   GitHub <https://github.com/selimfirat/pysad>
   Changelog <https://github.com/selimfirat/pysad/releases>
