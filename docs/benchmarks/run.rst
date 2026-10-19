Run the benchmark
=================

This page shows how to run the :doc:`multivariate detector benchmark </benchmark>` locally, update the tables for specific detectors and verify that a code change keeps ``docs/benchmark/`` up to date.

Run every detector
------------------

Install PySAD in editable mode, as :doc:`/contributing` shows. Then run the benchmark from the root of the repository:

.. code-block:: console

   $ python benchmarks/benchmark.py

The first run downloads the datasets into ``~/.cache/pysad``: the |num_datasets| ODDS datasets, the NAB and UCR series, and the streams with drift. The UCR archive (184 MB) and the KDD Cup 99 file (74 MB) are the largest downloads. The script writes the CSV tables to ``docs/benchmark/``, which the benchmark pages render. A full run takes about 50 minutes on 10 cores. The multivariate detectors on the NAB series take most of this time. The ``--jobs`` option sets the number of processes. The default is the number of CPU cores.

The ``--parts`` option runs only some parts of the benchmark, for example ``--parts drift``. These are the parts:

- ``odds``: the multivariate detectors on the shuffled ODDS datasets.
- ``univariate``: the univariate detectors on the NAB and UCR series.
- ``drift``: the multivariate detectors on the streams with drift.
- ``nab``: the multivariate detectors on the NAB series.
- ``shingles``: the multivariate detectors on shingles of the NAB series.

Each part runs once, with seed 0. The ``benchmark`` workflow runs each part in its own job.

Run selected detectors
----------------------

To run only some detectors again, give their class names with ``--detectors``. The script replaces the rows of those detectors in ``docs/benchmark/`` and keeps the other results:

.. code-block:: console

   $ python benchmarks/benchmark.py --detectors LODA,RSHash

For a change that targets speed, pass ``--jobs 1`` before and after the change so that the timings are comparable:

.. code-block:: console

   $ python benchmarks/benchmark.py --detectors RSHash --jobs 1

Commit your code changes before you run the benchmark. If you do not, the commit in the results ends with ``-dirty``.

Check against ``master``
------------------------

To list the detectors that a branch can affect and check whether their AUROC moved since ``origin/master``, run:

.. code-block:: console

   $ python benchmarks/benchmark.py --check origin/master

The ``benchmark`` workflow runs this check on each pull request that changes ``pysad/``. It runs again only the detectors that the change can affect, with one job per part. If an AUROC moves by more than 0.005, the check fails and prints the command that updates ``docs/benchmark/``. The check does not compare AUPR, because other platforms round differently. On a small dataset, such a difference can swap one anomaly with one normal point, and that moves AUPR much more than AUROC.
