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

pip installs these packages with PySAD. PySAD has no optional extras, so you do not install other packages. Every detector works after ``pip install pysad``. To run a PyOD detector through :mod:`pysad.models.integrations`, install PyOD with ``pip install pyod``.

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
     - Random state handling, random projections, random Fourier features, metrics and the Isolation Forest in :class:`~pysad.models.IForestASD`
   * - SciPy
     - 1.13.0
     - Statistical tests, distances, the error function and loading the benchmark datasets

Test the installation
---------------------

To test the installation, run this command. It prints the installed version:

.. code-block:: console

   $ python -c "import pysad; print(pysad.__version__)"
   0.6.0
