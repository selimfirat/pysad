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
* pyod==2.0.5
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



Contributors
============

.. image:: https://contrib.rocks/image?repo=selimfirat/pysad
    :target: https://github.com/selimfirat/pysad/graphs/contributors
    :alt: Contributors
    :align: center

We thank all our contributors for their valuable input and efforts to make PySAD better!

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