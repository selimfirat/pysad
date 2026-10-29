Run the benchmark
=================

This page shows how to run the :doc:`benchmark </benchmark>` on your computer, update the tables of some detectors, and compare a branch with ``master``.

Run every detector
------------------

Install PySAD in editable mode, as :doc:`/contributing` shows. Then run the benchmark from the root of the repository:

.. code-block:: console

   $ python benchmarks/benchmark.py

The first run downloads the datasets into ``~/.cache/pysad``: the |num_datasets| ODDS datasets, the NAB and UCR series, and the streams with drift. The UCR archive (184 MB) and the KDD Cup 99 file (74 MB) are the largest downloads. The script writes the CSV tables to ``docs/benchmark/``, which the benchmark pages render. A full run takes about 50 minutes on 10 cores. The multivariate detectors on the NAB series take most of this time. The ``--jobs`` option sets the number of processes. The default is the number of CPU cores. While the benchmark runs, it prints the number of runs that are done and the runs that remain, so that a slow run shows in the output.

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

Dimensionality sweep
--------------------

To measure how latency per point scales with the number of features, run the dimensionality sweep. It runs LODA, xStream, RSHash, HalfSpaceTrees, Inqmad, and ExactStorm on synthetic uniform streams at 2, 10, 50, 100, 500, and 1000 features, with every detector seeing the same stream at each dimension:

.. code-block:: console

   $ python benchmarks/benchmark.py --dimensions

The sweep needs no downloads. It writes ``docs/benchmark/dimensions.csv``, with one row per detector and dimension, and ``docs/benchmark/dimensions.svg``, with one scalability curve per detector. Each run warms up on 256 instances before timing. The ``--dimension-points`` option sets the measured points per detector and dimension, excluding warmup. The default is 2000.

Memory stability
----------------

To check that detector memory stays bounded over a long stream, run ``benchmarks/memory_stability.py``:

.. code-block:: console

   $ python benchmarks/memory_stability.py
   $ python benchmarks/memory_stability.py --detectors LODA,RSHash --points 100000 --checkpoint 10000

The script generates one point at a time, warms up on 1,000 points, and records ``tracemalloc`` memory every ``--checkpoint`` points (default 100,000) over ``--points`` points (default 1,000,000). It exits with a non-zero status if traced memory grows by more than ``--max-growth-kb`` KiB (default 512.0) between the second and last checkpoints.

Check against ``master``
------------------------

This command lists the detectors that a branch can affect. Then it runs them and compares their AUROC with ``docs/benchmark/``:

.. code-block:: console

   $ python benchmarks/benchmark.py --check origin/master

If an AUROC moves by more than 0.005, the check fails and prints the command that updates ``docs/benchmark/``. The check does not compare AUPR, because other platforms round differently. On a small dataset, such a difference can swap one anomaly with one normal point, and that moves AUPR much more than AUROC.

The short run in CI
-------------------

The ``benchmark`` workflow runs on each pull request that changes ``pysad/`` or ``benchmarks/``. It runs only the detectors that the pull request can change, in a short form. With ``--ci``, each run keeps only the first 2,000 points of its dataset, series or stream. The UCR series do not run. The parts ``odds``, ``univariate`` and ``drift`` each run in their own job. The parts ``nab`` and ``shingles`` do not run in CI. Each job does these steps:

#. It installs the code of the base branch and runs the detectors. ``--since`` picks the detectors from the files that changed:

   .. code-block:: console

      $ python benchmarks/benchmark.py --ci base.csv --since origin/master --parts odds

#. It installs the code of the pull request and runs the same detectors. ``--compare`` prints each run whose AUROC moved by more than 0.005:

   .. code-block:: console

      $ python benchmarks/benchmark.py --ci head.csv --since origin/master --parts odds --compare base.csv

#. It adds the runs that moved to the summary of the job, and it keeps ``base.csv`` and ``head.csv`` as an artifact.

Both runs use the same machine, so the results do not depend on the platform. The job does not fail when a result moves, because a pull request can change a detector on purpose. The full results in ``docs/benchmark/`` come from full runs that the maintainers make.
