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

   $ python benchmarks/benchmark.py --detectors RSHash --jobs 1 --no-cache

Commit your code changes before you run the benchmark. If you do not, the commit in the results ends with ``-dirty``.

Benchmark a custom model
------------------------

To benchmark a detector that is not part of PySAD, save it in a file and pass
``--custom path/to/file.py:ClassName``. The runner loads the class from that file and scores it on
the datasets whose part matches its input kind, the same way it scores the built-in detectors.

The class must subclass :class:`~pysad.core.BaseModel` and implement ``fit_partial`` and
``score_partial``, as :doc:`/guides/custom_model` shows.

A multivariate custom model is built once per dataset with ``factory(X, seed)``. The runner fills
these constructor arguments from the data when they are parameters: ``feature_mins``,
``feature_maxes``, ``input_shape`` and ``initial_X``. Give every other constructor argument a
default value. The runner passes ``seed`` to ``random_state`` when the constructor accepts it.

A univariate custom model sets ``_input_kind = "univariate"`` and is built with
``factory(n, period)``. The runner fills ``min_val``, ``max_val``, ``period``,
``window_length``, ``subsequence_length``, ``window_size`` and ``probationary_period`` when they
are parameters. Give every other constructor argument a default value. The runner passes
``random_state=0`` when the constructor accepts it.

For example, save this detector as ``my_model.py``:

.. code-block:: python

   import numpy as np
   from pysad.core import BaseModel


   class MyModel(BaseModel):
       """Scores each point by its distance from an exponentially weighted mean.

       Args:
           alpha (float): The weight of the newest point, between 0 and 1 (Default=0.05).
       """

       def __init__(self, alpha=0.05):
           self.alpha = alpha
           self.mean_ = None
           self.var_ = None

       def fit_partial(self, X, y=None):
           if self.mean_ is None:
               self.mean_ = X.astype(float)
               self.var_ = np.ones_like(self.mean_)
               return self

           diff = X - self.mean_
           self.mean_ += self.alpha * diff
           self.var_ = (1 - self.alpha) * (self.var_ + self.alpha * diff**2)
           return self

       def score_partial(self, X):
           if self.mean_ is None:
               return 0.0

           z = (X - self.mean_) / np.sqrt(self.var_ + 1e-12)
           return float(np.abs(z).max())

Then run the benchmark runner on this model:

.. code-block:: console

   $ python benchmarks/benchmark.py --custom my_model.py:MyModel

A multivariate model runs on the ODDS datasets and the streams with drift. A univariate model
runs on the NAB and UCR series. Custom models are not stored in the result cache, so each run
scores them again.

Benchmark an embedding stream
-----------------------------

To benchmark a frozen encoder, for example the output of a pretrained time-series foundation
model, save the encoder as a callable in a file and pass
``--encoder path/to/file.py:encode``. The script encodes every instance of a deterministic
synthetic stream, scores each embedding with
:class:`~pysad.models.ZeroShotEmbeddingDetector`, and reports the AUROC, the milliseconds per
point, and whether a second run on the same stream gives exactly the same scores.

The encoder receives one instance as a one dimensional NumPy array of shape
``(num_features,)`` and must return one embedding vector of shape ``(embedding_dim,)`` with the
same width on every call. It must be deterministic: if the encoder returns the same embedding for the same instance, a
second run gives exactly the same scores. No framework or new dependency is needed. The encoder
and its runtime stay outside PySAD.

The semantic anchors of the detector are the embeddings of the warm-up points, so no labels are
used to build them and the warm-up points do not count in the AUROC.

For example, save this encoder as ``my_encoder.py``:

.. code-block:: python

   import numpy as np


   def encode(x):
       x = np.asarray(x, dtype=np.float64)
       return x.copy()


Then run:

.. code-block:: console

   $ python benchmarks/bench_embedding_stream.py --encoder my_encoder.py:encode --points 2000 --seed 7

The command prints one row per run. If you pass ``--output results.csv``, the script writes the
results to that CSV file.

Result cache
------------

The script caches each detector-dataset result in ``~/.cache/pysad/benchmark``.
Unchanged pairs reuse their scores and timings.
The cache key includes source contents, detector settings, dataset names, Python, platform, and installed package versions.
Source contents include the detector, its PySAD imports, dataset definitions, preprocessing, calibrators, and the benchmark script.
Local source edits invalidate affected entries before you commit them.
Shingled runs and short CI runs use separate entries.
The cache keeps calibrated NAB scores so partial reruns can rebuild aggregate NAB scores.

Use ``--cache-dir PATH`` to choose another cache directory.
If you need fresh timings, pass ``--no-cache``.
This option skips cache reads and writes.
``--check`` always runs pairs again.
Custom models and dimensionality sweeps also run without this result cache.
The separate CI base cache still uses ``--base-cache-dir`` and ``--det-hashes``.
To clear cached pair results, delete their cache directory.

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

Edge-memory and throughput benchmark
------------------------------------

To measure the peak traced memory and the milliseconds per point of the pure-Python streaming
detectors on a deterministic stream of a fixed length, run ``benchmarks/bench_stream.py``:

.. code-block:: console

   $ python benchmarks/bench_stream.py --detectors all --points 10000 --repeats 3
   $ python benchmarks/bench_stream.py --detectors LODA,RSHash,MIDAS --points 50000 --output docs/benchmark/stream_benchmark.csv

Each detector warms up on 64 points, then scores every point before it learns from it. The stream
length and the seed are fixed, so a rerun gives the same memory and timing. tracemalloc records
the peak traced memory, and the script reports the median milliseconds per point across the timed
runs. Multivariate detectors see five features from the unit interval, univariate detectors see
one value, timestamped detectors see ``(timestamp, value)``, and edge-stream detectors see
``(source, destination)``.

Reproducible drift benchmark (TSB-drift)
----------------------------------------

To benchmark the public detectors on the real streams with drift in a reproducible way, run
``benchmarks/bench_tsb_drift.py``:

.. code-block:: console

   $ python benchmarks/bench_tsb_drift.py --points 10000 --seed 0 --repeats 3 --output drift.csv

The script loads each stream with drift (KDD http, INSECTS A/I/IG/IR, SMD 1-1/2-1/3-1), keeps at
most ``--points`` points in time order, scales each feature to the range from 0 to 1, and scores
every public multivariate detector with ``fit_score_partial``. It reports streaming AUROC and AUPR
after the warm-up, milliseconds per point (throughput), and traced memory from ``tracemalloc``. A
second run on the same stream must give exactly the same scores, so the run is reproducible.

Use ``--detectors`` to score only some detectors and ``--streams`` to score only some streams. The
results go to the file given with ``--output``; without it, the script prints a table only.

Streaming detector ensembles
----------------------------

To report prequential (test-then-train) metrics of streaming detector ensembles, run the
``--ensembles`` mode:

.. code-block:: console

   $ python benchmarks/benchmark.py --ensembles
   $ python benchmarks/benchmark.py --ensembles --ci --jobs 1

Each ensemble combines the same fast multivariate detectors (LODA, xStream, RSHash, and
HalfSpaceTrees) with a different score ensembler: average, maximum, median, average of the bucket
maxima, or maximum of the bucket averages. Every detector is wrapped in a normalizing pipeline so
that its scores are on a comparable scale before they are combined.

The mode scores every instance of each ODDS dataset before the ensemble learns from it, using
:func:`~pysad.evaluation.evaluate_stream`, and reports AUROC, AUPR, and milliseconds per point.
With ``--ci`` it keeps only the first 2,000 points of each dataset. The results go to
``docs/benchmark/ensemble_results.csv``, one row per dataset and ensemble.

Performance badges
------------------

The README and documentation home page read badge values from ``docs/_static/*.json``.
Throughput uses the median points per second across valid, positive timings in ``docs/benchmark/results.csv``.
Detector counts use the public catalog and exclude helper models.
Memory uses the largest traced growth in ``docs/benchmark/memory.csv``.
The recorded memory measurements cover three detectors over 1,000,000 points after warmup.
They do not measure total process memory or all detectors.

If you update the memory measurements, write the CSV with this command:

.. code-block:: console

   $ python benchmarks/memory_stability.py --output docs/benchmark/memory.csv

If you update benchmark results or the detector catalog, regenerate the badges:

.. code-block:: console

   $ python scripts/generate_performance_badges.py

Commit the generated JSON files with their sources. The docs workflow rejects stale values.

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
