Benchmark
=========

This page compares the multivariate detectors of PySAD on 17 labeled datasets. A multivariate detector takes a vector of features for each point. Each detector uses its default parameters. Use the results to find two or three candidates. Then compare the candidates on your own data, as :doc:`guides/choose_detector` shows.

Results
-------

The benchmark measures how well the scores of a detector separate the anomalies from the normal points. It uses two metrics:

AUROC
   The probability that a random anomaly gets a higher score than a random normal point. A random detector gets 0.5. A perfect detector gets 1.

AUPR
   The average precision over all the score thresholds. A random detector gets the fraction of anomalies in the dataset. A perfect detector gets 1.

:doc:`guides/evaluate` explains the two metrics in more detail.

Summary
~~~~~~~

The summary has one row for each detector, from the best mean rank to the worst. It has these columns:

Mean rank
   The rank of a detector on a dataset is its position by AUROC, where 1 is the highest AUROC. The mean rank is the average of these ranks over the 17 datasets.

Mean AUROC and Mean AUPR
   The averages over the 17 datasets and the 5 seeds.

Time per point
   The median time of one ``fit_score_partial`` call, in milliseconds. The runs shared one machine, so compare these times only with each other.

Commit
   The commit of PySAD that gave the results. Click it to see the code on GitHub.

.. csv-table::
   :file: benchmark/summary.csv
   :header-rows: 1
   :class: benchmark-table

AUROC on each dataset
~~~~~~~~~~~~~~~~~~~~~

Each value is the mean over the 5 seeds. The highest value on each dataset is in bold. The columns are in the order of the summary. The table uses three short names: HST is :class:`~pysad.models.HalfSpaceTrees`, LoOP is :class:`~pysad.models.LocalOutlierProbability`, and RRCF is :class:`~pysad.models.RobustRandomCutForest`.

.. csv-table::
   :file: benchmark/auroc.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

AUPR on each dataset
~~~~~~~~~~~~~~~~~~~~

This table has the same layout as the AUROC table. The AUPR of a random detector is different on each dataset, so compare the values only in the same row.

.. csv-table::
   :file: benchmark/aupr.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

How the benchmark works
-----------------------

The benchmark follows the rules of :doc:`guides/evaluate`. For each dataset, each detector and each seed from 0 to 4, it does these steps:

#. It loads the dataset with :class:`~pysad.utils.Data`. Then it shuffles the points with ``ArrayStreamer(shuffle=True, random_state=seed)``. Many of the files put all the anomalies at the end, so the order of the file is not a fair stream.
#. It keeps the first 5,000 points of the shuffled stream.
#. It scales each feature to the range from 0 to 1, with the minimum and the maximum from :func:`~pysad.utils.get_minmax_array`. The default parameters of some detectors, for example the radius of :class:`~pysad.models.ExactStorm`, assume features of about this size.
#. It makes the detector with its default parameters and ``random_state=seed``. Some detectors also get the arguments that they require. HST and :class:`~pysad.models.RSHash` get the range from 0 to 1 for each feature. :class:`~pysad.models.Inqmad` gets the number of features. LoOP fits on the first 256 points of the stream.
#. It gives each point to ``fit_score_partial``, which trains the detector on the point and scores the point.
#. It computes AUROC and AUPR over the scores of all the points.

Step 3 uses the minimum and the maximum of the whole stream. A live stream does not know them in advance. For a live stream, take them from historical data, or use known physical limits.

Datasets
~~~~~~~~

The datasets come from the ODDS repository (Outlier Detection DataSets). The column "Points used" gives the number of points after the limit of 5,000.

.. csv-table::
   :file: benchmark/datasets.csv
   :header-rows: 1
   :stub-columns: 1
   :class: benchmark-table

What the results do not show
----------------------------

- The univariate detectors take one value per point, so the benchmark does not run them. These detectors are :class:`~pysad.models.MedianAbsoluteDeviation`, :class:`~pysad.models.StandardAbsoluteDeviation`, :class:`~pysad.models.KNNCAD`, :class:`~pysad.models.RelativeEntropy`, :class:`~pysad.models.SeasonalESD` and :class:`~pysad.models.SeasonalHybridESD`.
- :class:`~pysad.models.integrations.ReferenceWindowModel` runs a PyOD model, so its results depend on that model. :class:`~pysad.models.IForestASD` is one example of it.
- The points of these datasets are independent, and the benchmark puts them in a random order. The streams have no drift and no seasonality. Thus the results do not show how a detector follows changes over time.
- Tuned parameters can give very different results. For example, the default radius of :class:`~pysad.models.ExactStorm` finds no neighbors on most of these datasets, so its AUROC is near 0.5. :doc:`guides/choose_detector` names the parameters with the largest effect.

Run the benchmark
-----------------

Install PySAD in editable mode, as :doc:`contributing` shows. Then run the benchmark from the root of the repository:

.. code-block:: console

   $ python benchmarks/benchmark.py

The first run downloads the datasets. The script writes the results to ``docs/benchmark/``, and this page shows them. A full run takes about 4.5 CPU-hours, which is 30 minutes on 10 cores. The ``--jobs`` option sets the number of processes. The default is the number of CPU cores.

To run some detectors again, give their names with ``--detectors``. The script replaces the results of these detectors and keeps the other results:

.. code-block:: console

   $ python benchmarks/benchmark.py --detectors LODA,RSHash

Commit your code changes before you run the benchmark. If you do not, the commit in the results ends with ``-dirty``.

The ``benchmark`` workflow runs on each pull request that changes ``pysad/``. It runs again only the detectors that the change can affect. If a result moves by more than 0.002, the check fails. The output of the check gives the command that updates ``docs/benchmark/``.
