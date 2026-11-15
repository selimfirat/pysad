Benchmark results
=================

The benchmark runs the detectors of PySAD on labeled data and compares their results. Each detector runs once on each dataset, with its default parameters and seed 0. Use the results to find two or three candidates for your data. Then compare these candidates on your own data, as :doc:`guides/choose_detector` shows.

A multivariate detector takes a vector of features for each point. A univariate detector takes one value for each point. See :doc:`benchmarks/methodology` for the evaluation procedure and its limits.

The results are on these pages:

- :doc:`benchmarks/multivariate`: the multivariate detectors on |num_datasets| ODDS datasets and on |num_drift_streams| real streams with drift.
- :doc:`benchmarks/univariate`: the univariate detectors on the NAB and UCR series, and the multivariate detectors on the NAB series.
- :doc:`benchmarks/calibration`: the alerts of the two probability calibrators.

Benchmark setup
---------------

- :doc:`benchmarks/methodology`: evaluation steps, metrics and limitations.
- :doc:`benchmarks/datasets`: dataset sources and stream construction.
- :doc:`benchmarks/run`: local runs, result updates and CI comparisons.

.. toctree::
   :hidden:

   benchmarks/multivariate
   benchmarks/univariate
   benchmarks/calibration
   benchmarks/methodology
   benchmarks/datasets
   benchmarks/run
