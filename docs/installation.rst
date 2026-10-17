Installation
============

PySAD supports Python 3.10 to 3.14 on Linux, macOS and Windows.

Install from PyPI
-----------------

.. code-block:: console

   $ pip install pysad

To upgrade an existing installation, add ``-U``:

.. code-block:: console

   $ pip install -U pysad

Install from source
-------------------

To use changes that are not released yet, install from the ``master`` branch on GitHub:

.. code-block:: console

   $ pip install git+https://github.com/selimfirat/pysad.git

To change the code, see :doc:`contributing`.

Dependencies
------------

pip installs these packages with PySAD. PySAD has no optional extras: every detector works after ``pip install pysad``.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Package
     - Minimum version
     - Used for
   * - NumPy
     - 1.22.4
     - Arrays in every component
   * - scikit-learn
     - 1.4.2
     - Random state handling, random projections and metrics
   * - SciPy
     - 1.13.0
     - Statistical tests and distributions
   * - statsmodels
     - 0.14.2
     - STL decomposition in :class:`~pysad.models.SeasonalESD`, :class:`~pysad.models.SeasonalHybridESD` and :class:`~pysad.transform.preprocessing.ModifiedSTLResidualTransformer`
   * - PyOD
     - 3.6.2
     - Batch detectors for :mod:`pysad.models.integrations` and :class:`~pysad.models.IForestASD`
   * - pandas
     - 2.2.2
     - :class:`~pysad.utils.PandasStreamer`
   * - rrcf
     - 0.4.4
     - :class:`~pysad.models.RobustRandomCutForest`
   * - PyNomaly
     - 0.4.0
     - :class:`~pysad.models.LocalOutlierProbability`

Make sure that the installation works
-------------------------------------

Run this command. It prints the installed version:

.. code-block:: console

   $ python -c "import pysad; print(pysad.__version__)"
