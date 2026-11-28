Optional PyO3 traversal kernels
===============================

PySAD ships with optional PyO3 Rust kernels via the ``pysad[kernels]`` extra. These kernels speed up the tree traversal hot path of two detectors:

- :class:`~pysad.models.HalfSpaceTrees` routes each point down its trees.
- :class:`~pysad.models.RobustRandomCutForest` routes each point to a leaf and computes CoDisp.

Without the extension, both detectors use NumPy. The benchmarks in ``benchmarks/bench_hst_kernels.py`` and ``benchmarks/bench_rrcf_kernels.py`` time the traversal only. They do not time tree updates or anomaly scoring.

Install the extension to run the Rust path:

.. code-block:: console

   $ pip install "pysad[kernels]"

Then run the benchmarks. Each script compares NumPy with Rust and checks that both paths give the same result:

.. code-block:: console

   $ python benchmarks/bench_hst_kernels.py
   $ python benchmarks/bench_rrcf_kernels.py

Without the extension, run the NumPy path only:

.. code-block:: console

   $ python benchmarks/bench_hst_kernels.py --numpy-only
   $ python benchmarks/bench_rrcf_kernels.py --numpy-only

Results
-------

The table shows the NumPy traversal time per point, measured on this machine. Your numbers depend on your CPU and Python build. Each run uses 1000 points per timing run and 10 repeats.

+----------------------------------+-------------------+------------------------+
| Benchmark                        | Configuration     | NumPy time per point   |
+==================================+===================+========================+
| HalfSpaceTrees path traversal    | 10 features       | 0.030 ms               |
|                                  | 25 trees          |                        |
|                                  | max depth 15      |                        |
+----------------------------------+-------------------+------------------------+
| RobustRandomCutForest route to   | 10 features       | 0.183 ms               |
| leaf and CoDisp                  | 50 trees          |                        |
|                                  | tree size 256     |                        |
+----------------------------------+-------------------+------------------------+

The PyO3 kernels replace the NumPy loops with Rust code. When ``pysad[kernels]`` is installed, the detectors use the Rust path automatically. The same benchmarks then print the Rust time and the speedup over NumPy.

Traversal cost stays bounded. The cost of routing one point does not grow with the stream length, because each tree holds a fixed window of instances.
