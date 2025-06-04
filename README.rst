.. image:: https://github.com/selimfirat/pysad/raw/master/docs/logo.png
    :align: center

Python Streaming Anomaly Detection (PySAD)
==========================================

.. image:: https://img.shields.io/pypi/v/pysad
    :target: https://pypi.org/project/pysad/
    :alt: PyPI

.. image:: https://img.shields.io/github/v/release/selimfirat/pysad
   :target: https://github.com/selimfirat/pysad/releases
   :alt: GitHub release (latest by date)

.. image:: https://readthedocs.org/projects/pysad/badge/?version=latest
   :target: https://pysad.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status

.. image:: https://badges.gitter.im/selimfirat-pysad/community.svg
   :target: https://gitter.im/selimfirat-pysad/community?utm_source=share-link&utm_medium=link&utm_campaign=share-link
   :alt: Gitter

.. image:: https://dev.azure.com/selimfirat/pysad/_apis/build/status/selimfirat.pysad?branchName=master
   :target: https://dev.azure.com/selimfirat/pysad/_build/latest?definitionId=2&branchName=master
   :alt: Azure Pipelines Build Status

.. image:: https://travis-ci.org/selimfirat/pysad.svg?branch=master
   :target: https://travis-ci.org/selimfirat/pysad
   :alt: Travis CI Build Status

.. image:: https://ci.appveyor.com/api/projects/status/ceghuv517ghqgjce/branch/master?svg=true
   :target: https://ci.appveyor.com/project/selimfirat/pysad/branch/master
   :alt: Appveyor Build status

.. image:: https://circleci.com/gh/selimfirat/pysad.svg?style=svg
   :target: https://circleci.com/gh/selimfirat/pysad
   :alt: Circle CI

.. image:: https://coveralls.io/repos/github/selimfirat/pysad/badge.svg?branch=master
   :target: https://coveralls.io/github/selimfirat/pysad?branch=master
   :alt: Coverage Status

.. image:: https://img.shields.io/pypi/pyversions/pysad
   :target: https://github.com/selimfirat/pysad/
   :alt: PyPI - Python Version

.. image:: https://img.shields.io/badge/platforms-linux--64%2Cosx--64%2Cwin--64-green
   :target: https://github.com/selimfirat/pysad/
   :alt: Supported Platforms

.. image:: https://img.shields.io/github/license/selimfirat/pysad.svg
   :target: https://github.com/selimfirat/pysad/blob/master/LICENSE
   :alt: License


**PySAD** is an open-source python framework for anomaly detection on streaming multivariate data.

`Documentation <https://pysad.readthedocs.io/en/latest/>`__

Features
========

Online Anomaly Detection
^^^^^^^^^^^^^^^^^^^^^^^^

`PySAD` provides methods for online/sequential anomaly detection, i.e. anomaly detection on streaming data, where model updates itself as a new instance arrives.


Resource-Efficient
^^^^^^^^^^^^^^^^^^

Streaming methods efficiently handle the limitied memory and processing time requirements of the data streams so that they can be used in near real-time. The methods can only store an instance or a small window of recent instances.


Complete
^^^^^^^^

`PySAD` contains stream simulators, evaluators, preprocessors, statistic trackers, postprocessors, probability calibrators and more. In addition to streaming models, `PySAD` also provides integrations for batch anomaly detectors of the `PyOD <https://github.com/yzhao062/pyod/>`_ so that they can be used in the streaming setting.


Comprehensive
^^^^^^^^^^^^^

`PySAD` serves models that are specifically designed for both univariate and multivariate data. Furthermore, one can experiment via `PySAD` in supervised, semi-supervised and unsupervised setting.


User Friendly
^^^^^^^^^^^^^

Users with any experience level can easily use `PySAD`. One can easily design experiments and combine the tools in the framework. Moreover, the existing methods in `PySAD` are easy to extend.


Free and Open Source Software (FOSS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`PySAD` is distributed under `BSD License 2.0 <https://github.com/selimfirat/pysad/blob/master/LICENSE>`_ and favors FOSS principles.

Community Engagement
=====================

PySAD has built a strong and active community with significant adoption across academia and industry:

Academic Recognition
^^^^^^^^^^^^^^^^^^^^

* **Cited in academic literature** with growing adoption in streaming data research with more than 50 citations to the arXiv version (excluding GitHub link-only citations). See `Google Scholar <https://scholar.google.com/citations?view_op=view_citation&hl=tr&user=R6Hwp20AAAAJ&citation_for_view=R6Hwp20AAAAJ:2osOgNQ5qMEC>`_ for detailed citation metrics.

GitHub Community
^^^^^^^^^^^^^^^^^

* **260+ GitHub Stars** demonstrating widespread community interest and adoption among developers and researchers in the machine learning community.

Active Usage
^^^^^^^^^^^^

* **Strong PyPI download statistics** according to `pypistats.org <https://pypistats.org/packages/pysad>`_ with 2K+ downloads in the May 2025 and consistent weekly usage.

Educational Content
^^^^^^^^^^^^^^^^^^^

Featured in educational content across multiple platforms:

* **Medium Articles:**

  * `Real-time Anomaly Detection with Python <https://medium.com/data-science/real-time-anomaly-detection-with-python-36e3455e84e2>`_
  * `Real-time Anomaly Detection for Quality Control <https://medium.com/data-science/real-time-anomaly-detection-for-quality-control-e6af28a3350d>`_
  * `The Challenges of AI in an Industrial Environment <https://medium.com/@anthonycvn/the-challenges-of-ai-in-an-industrial-environment-6e118a8daa67>`_

* **Resource Collections:**

  * `Anomaly Detection Resources <https://andrewm4894.com/2021/01/03/anomaly-detection-resources/>`_ comprehensive guide
  * `Anomaly Detection Resources by Yue Zhao <https://github.com/yzhao062/anomaly-detection-resources>`_ curated collection of papers, algorithms, and datasets

Third-party Integrations
^^^^^^^^^^^^^^^^^^^^^^^^^

PySAD has been adopted and integrated into major machine learning frameworks:

* **TurboML Integration:** `PySAD example documentation <https://docs.turboml.com/wyo_models/pysad_example/>`_ showing adoption in machine learning workflow platforms.

* **Apache Beam Integration:** PySAD modules adapted into Apache Beam's ML package with `zscore <https://beam.apache.org/releases/pydoc/2.64.0/apache_beam.ml.anomaly.detectors.zscore.html>`_ and `robust_zscore <https://beam.apache.org/releases/pydoc/2.64.0/apache_beam.ml.anomaly.detectors.robust_zscore.html>`_ anomaly detectors.

* **River ML Integration:** The prominent online machine learning library has adapted PySAD algorithms, including the `StandardAbsoluteDeviation detector <https://riverml.xyz/0.20.0/api/anomaly/StandardAbsoluteDeviation/?query=pysad>`_ with explicit PySAD references.

Developer Community
^^^^^^^^^^^^^^^^^^^

* **Widespread GitHub usage** with 50+ files using ``import pysad`` and 200+ files using ``from pysad`` across various repositories: `import usage <https://github.com/search?q=%22import+pysad%22&type=code>`_, `from usage <https://github.com/search?q=%22from+pysad%22&type=code>`_.

* **External projects** demonstrating practical applications across diverse domains:

  * `Online Isolation Forest implementation <https://github.com/ineveLoppiliF/Online-Isolation-Forest>`_
  * `Anomaly Detection Final Project <https://github.com/berfinkavsut/anomaly-detection-final-project>`_
  * `Natural Gas Wells Anomaly Detection <https://github.com/charles-cao/Anomaly-detection-in-natural-gas-wells>`_
  * `EuroPython 2024 Conference Material <https://github.com/ciortanmadalina/europython2024>`_

Installation
============


The PySAD framework can be installed via:


.. code-block:: bash

    pip install -U pysad


Alternatively, you can install the library directly using the source code in Github repository by:


.. code-block:: bash

    git clone https://github.com/selimfirat/pysad.git
    cd pysad
    pip install .


**Required Dependencies:**


* Python 3.10+
* numpy==2.0.2
* scikit-learn==1.5.2
* scipy==1.13.1
* pyod==1.1.0
* combo==0.1.3

**Optional Dependencies:**


* rrcf==0.4.3 (Only required for  `pysad.models.robust_random_cut_forest.RobustRandomCutForest`)
* PyNomaly==0.3.3 (Only required for  `pysad.models.loop.StreamLocalOutlierProbability`)
* mmh3==2.5.1 (Only required for  `pysad.models.xstream.xStream`)
* pandas==2.2.3 (Only required for  `pysad.utils.pandas_streamer.PandasStreamer`)
* jax>=0.6.1 (Only required for  `pysad.models.inqmad.Inqmad`; required for NumPy 2.0+ compatibility)
* jaxlib>=0.6.1 (Only required for  `pysad.models.inqmad.Inqmad`; required for NumPy 2.0+ compatibility)

Examples
========

Quick Start
^^^^^^^^^^^^^^^^^^

Here's a simple example showing how to use PySAD for anomaly detection on streaming data:

.. code-block:: python

    # Import modules.
    from pysad.evaluation import AUROCMetric
    from pysad.models import LODA
    from pysad.utils import Data


    model = LODA()  # Init model
    metric = AUROCMetric()  # Init area under receiver-operating- characteristics curve metric
    streaming_data = Data().get_iterator("arrhythmia.mat")  # Get data streamer.

    for x, y_true in streaming_data:  # Stream data.
        anomaly_score = model.fit_score_partial(x)  # Fit the instance to model and score the instance.

        metric.update(y_true, anomaly_score)  # Update the AUROC metric.

    # Output the resulting AUROCMetric.
    print(f"Area under ROC metric is {metric.get()}.")

Quick Links
============

* `Github Repository <https://github.com/selimfirat/pysad/>`_

* `Documentation <http://pysad.readthedocs.io/>`__

* `PyPI Package <https://pypi.org/project/pysad>`_

* `Travis CI <https://travis-ci.com/github/selimfirat/pysad>`_

* `Azure Pipelines <https://dev.azure.com/selimfirat/pysad/>`_

* `Circle CI <https://circleci.com/gh/selimfirat/pysad/>`_

* `Appveyor <https://ci.appveyor.com/project/selimfirat/pysad/branch/master>`_

* `Coveralls <https://coveralls.io/github/selimfirat/pysad?branch=master>`_

* `License <https://github.com/selimfirat/pysad/blob/master/LICENSE>`_



Versioning
==========

`Semantic versioning <http://semver.org/>`_ is used for this project.

License
=======

This project is licensed under the `BSD License 2.0 <https://github.com/selimfirat/pysad/blob/master/LICENSE>`_.


Citing PySAD
============
If you use PySAD for a scientific publication, please cite the following paper:

.. code-block::

    @article{pysad,
      title={PySAD: A Streaming Anomaly Detection Framework in Python},
      author={Yilmaz, Selim F and Kozat, Suleyman S},
      journal={arXiv preprint arXiv:2009.02572},
      year={2020}
    }