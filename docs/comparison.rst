Comparison with River and PyOD
==============================

This page compares the multivariate detectors of PySAD with the streaming detectors of `River <https://riverml.xyz/>`_ and with batch detectors of `PyOD <https://github.com/yzhao062/pyod>`_. Each detector runs once on each of the |num_datasets| ODDS datasets of the :doc:`benchmark <benchmark>`, with the same points and seed 0.

Each detector scores each point before it learns from the point, as :doc:`benchmarks/methodology` describes. The metrics leave out the scores of the warm-up points, the first 256 points of a stream. The detectors use these settings:

- The PySAD detectors use the settings of the benchmark.
- The River detectors use their default parameters. Their points are dictionaries of the features, and the times do not include the conversion of the points to dictionaries.
- The PyOD detectors run in :class:`~pysad.models.integrations.ReferenceWindowModel` with their default parameters. Each detector first trains on the warm-up points. Then, after each 100 points, it trains a new detector on the last 1,000 points.

The table has one row for each detector, from the best mean rank to the worst. The columns are the same as in the summary of :doc:`benchmarks/multivariate`, and the best value of each column is in bold. The first column gives the version of each library that gave the results. For PySAD, the version also gives the commit of the run, because the run uses the code of the repository.

.. csv-table::
   :file: benchmark/comparison.csv
   :header-rows: 1
   :class: benchmark-table best-columns

All the runs used one computer, so compare the times only with each other. The time of a PyOD detector includes its training runs. A point that starts a training run takes much longer than the other points.

River's ``HalfSpaceTrees`` gives every point the score 0 until its first window of 250 points is full. Thus, on the datasets with fewer than about 500 points, it gets few or no informative scores after the warm-up.

River recommends that you scale the data of ``OneClassSVM`` with ``StandardScaler`` and add features with ``RBFSampler``. This comparison uses the defaults of all the detectors, so ``OneClassSVM`` gets the points without these steps.

To run the comparison, install the ``compare`` dependency group, which needs Python 3.11 or newer. Then run the script from the root of the repository:

.. code-block:: console

   $ pip install -e . --group compare
   $ python benchmarks/compare.py

The script writes ``docs/benchmark/comparison.csv``. With the same versions of the libraries, a new run gives the same AUROC and AUPR. Only the times change.
