Benchmark overview
==================

The benchmark runs the detectors of PySAD on labeled data and compares their results. Use the results to find two or three candidates for your data. Then compare these candidates on your own data, as :doc:`guides/choose_detector` shows.

A multivariate detector takes a vector of features for each point. A univariate detector takes one value for each point. The benchmark has five parts:

.. list-table::
   :header-rows: 1

   * - Part
     - Detectors
     - Data
     - Results
   * - ``odds``
     - Multivariate
     - |num_datasets| ODDS datasets, with the points in a random order
     - :doc:`benchmarks/multivariate`
   * - ``drift``
     - Multivariate
     - 8 real streams with drift, with the points in their time order
     - :doc:`benchmarks/drift`
   * - ``univariate``
     - Univariate
     - 58 NAB series and 84 UCR series, in their time order
     - :doc:`benchmarks/univariate`
   * - ``nab`` and ``shingles``
     - Multivariate
     - The 58 NAB series, as single values and as shingles of 10 values
     - :doc:`benchmarks/univariate`

Each part runs each detector once on each dataset, with the default parameters of the detector and seed 0. The detector scores each point before it learns from it. :doc:`benchmarks/calibration` compares the two probability calibrators on the runs of the ``odds`` part.

:doc:`benchmarks/methodology` explains the steps of a run, the metrics and the limits of the results. :doc:`benchmarks/datasets` describes the data. :doc:`benchmarks/run` shows how to run the benchmark on your computer.
