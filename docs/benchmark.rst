Benchmark results
=================

The benchmark runs the detectors of PySAD on labeled data and compares their results. Each detector runs once on each dataset, with its default parameters and seed 0. Use the results to find two or three candidates for your data. Then compare these candidates on your own data, as :doc:`guides/choose_detector` shows.

A multivariate detector takes a vector of features for each point. A univariate detector takes one value for each point. :doc:`benchmarks/methodology` explains the steps of a run, the metrics, the datasets and the limits of the results.

The results are on these pages:

- :doc:`benchmarks/multivariate`: the multivariate detectors on |num_datasets| ODDS datasets and on 8 real streams with drift.
- :doc:`benchmarks/univariate`: the univariate detectors on the NAB and UCR series, and the multivariate detectors on the NAB series.
- :doc:`benchmarks/calibration`: the alerts of the two probability calibrators.

.. toctree::
   :hidden:

   benchmarks/multivariate
   benchmarks/univariate
   benchmarks/calibration
