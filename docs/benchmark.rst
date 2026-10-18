Benchmark
=========

This page compares the multivariate detectors of PySAD on the 17 labeled datasets of :class:`~pysad.utils.Data`. Each detector uses its default parameters. Use the results to find two or three candidates. Then compare the candidates on your own data, as :doc:`guides/choose_detector` shows.

Results
-------

The rank of a detector on a dataset is its position by AUROC, where 1 is the highest AUROC. The mean rank is the average of these ranks over the 17 datasets. The mean AUROC and the mean AUPR are averages over the datasets and the 5 seeds. :doc:`guides/evaluate` explains the two metrics.

.. csv-table::
   :file: benchmark/summary.csv
   :header-rows: 1

The time per point is the median time of one ``fit_score_partial`` call over all the runs of the detector. It comes from one machine that ran several detectors at the same time, so compare the times only with each other.

AUROC on each dataset
~~~~~~~~~~~~~~~~~~~~~

Each value is the mean over 5 seeds.

.. csv-table::
   :file: benchmark/auroc.csv
   :header-rows: 1
   :stub-columns: 1

AUPR on each dataset
~~~~~~~~~~~~~~~~~~~~

Each value is the mean over 5 seeds. A random detector gets the fraction of anomalies in the dataset, so compare AUPR only on the same dataset.

.. csv-table::
   :file: benchmark/aupr.csv
   :header-rows: 1
   :stub-columns: 1

How the benchmark works
-----------------------

The benchmark follows the rules of :doc:`guides/evaluate`. For each dataset, detector and seed from 0 to 4, it does these steps:

#. It loads the dataset with :class:`~pysad.utils.Data` and shuffles it with ``ArrayStreamer(shuffle=True, random_state=seed)``. Many of the files put all anomalies at the end, so the order of the file is not a fair stream.
#. It keeps the first 5,000 points of the shuffled stream. The table below gives the number of points of each dataset.
#. It scales each feature to the range from 0 to 1 with the minimum and the maximum from :func:`~pysad.utils.get_minmax_array`. The default parameters of some detectors, for example the radius of :class:`~pysad.models.ExactStorm`, assume features of about this size.
#. It makes the detector with its default parameters and ``random_state=seed``. A detector also gets the arguments that it requires. :class:`~pysad.models.HalfSpaceTrees` and :class:`~pysad.models.RSHash` get the range from 0 to 1 for each feature. :class:`~pysad.models.Inqmad` gets the number of features. :class:`~pysad.models.LocalOutlierProbability` fits on the first 256 points of the stream.
#. It gives each point to ``fit_score_partial``, which uses the evaluation order from the paper of the detector.
#. It computes AUROC and AUPR over the scores of all the points.

The scaling uses the minimum and the maximum of the whole stream. A live stream does not know them in advance. For a live stream, take them from historical data, or use known physical limits.

.. csv-table::
   :file: benchmark/datasets.csv
   :header-rows: 1
   :stub-columns: 1

What the results do not show
----------------------------

- The univariate detectors take one value per point, so the benchmark does not run them. These detectors are :class:`~pysad.models.MedianAbsoluteDeviation`, :class:`~pysad.models.StandardAbsoluteDeviation`, :class:`~pysad.models.KNNCAD`, :class:`~pysad.models.RelativeEntropy`, :class:`~pysad.models.SeasonalESD` and :class:`~pysad.models.SeasonalHybridESD`.
- :class:`~pysad.models.integrations.ReferenceWindowModel` runs a PyOD model, so its results depend on that model. :class:`~pysad.models.IForestASD` is one example of it.
- The datasets are tables of independent points in a random order. They have no drift and no seasonality, so the results do not show how a detector follows changes over time.
- Tuned parameters can give very different results. For example, the default radius of :class:`~pysad.models.ExactStorm` finds no neighbors on most of these datasets, so its AUROC is near 0.5. :doc:`guides/choose_detector` names the parameters with the largest effect.

Run the benchmark
-----------------

Run the benchmark from the repository root, after you install PySAD in editable mode (see :doc:`contributing`). The first run downloads the datasets.

.. code-block:: console

   $ python benchmarks/benchmark.py

The script writes the results to ``docs/benchmark/``, which this page shows. A full run takes about 4.5 CPU-hours, or 30 minutes on 10 cores. ``--jobs`` sets the number of processes, which is the number of CPU cores by default.

To run some detectors again, give their names. The script replaces the rows of these detectors and keeps the other rows:

.. code-block:: console

   $ python benchmarks/benchmark.py --detectors LODA,RSHash

The ``benchmark`` workflow runs on each pull request that changes ``pysad/``. It runs again only the detectors that the change can affect. If a result moves by more than 0.002, the check fails and gives the command that updates ``docs/benchmark/``.
