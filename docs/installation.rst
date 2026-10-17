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

To use the newest unreleased changes, install from the ``master`` branch on GitHub:

.. code-block:: console

   $ pip install git+https://github.com/selimfirat/pysad.git

To change the code, see :doc:`contributing`.

Dependencies
------------

pip installs these packages with PySAD. PySAD has no optional extras, so you do not install other packages. Every detector works after ``pip install pysad``.

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
     - Random state handling, random projections, random Fourier features and metrics
   * - SciPy
     - 1.13.0
     - Statistical tests, distances and loading the benchmark datasets
   * - statsmodels
     - 0.14.2
     - STL decomposition (split of a series into trend, cycle and remainder) in :class:`~pysad.models.SeasonalESD`, :class:`~pysad.models.SeasonalHybridESD` and :class:`~pysad.transform.preprocessing.ModifiedSTLResidualTransformer`
   * - PyOD
     - 3.6.2
     - Batch detectors for :mod:`pysad.models.integrations` and :class:`~pysad.models.IForestASD`, and parameter checks in the ensemblers
   * - pandas
     - 2.2.2
     - DataFrames for :class:`~pysad.utils.PandasStreamer`
   * - rrcf
     - 0.4.4
     - :class:`~pysad.models.RobustRandomCutForest`
   * - PyNomaly
     - 0.4.0
     - :class:`~pysad.models.LocalOutlierProbability`

Test the installation
---------------------

To test the installation, run this command. It prints the installed version:

.. code-block:: console

   $ python -c "import pysad; print(pysad.__version__)"
   0.6.0
