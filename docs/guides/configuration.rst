.. meta::
   :description: Configure pysad-live, the pysad.mcp server, and the optional PyO3 kernels.

Configure PySAD
===============

This guide brings together configuration for three parts of PySAD: the ``pysad-live`` command, the ``pysad.mcp`` server, and the optional PyO3 kernels. Each section links to the guide with the full details.

pysad-live
----------

Install PySAD to get the ``pysad-live`` command:

.. code-block:: console

   $ pip install pysad

Run it without options to score the built-in synthetic stream with LODA in the terminal:

.. code-block:: console

   $ pysad-live

The command groups options into data source, detector, calibrator, display, web dashboard, and output.

Data source
~~~~~~~~~~~

.. code-block:: console

   # Built-in streams
   $ pysad-live --source nyc_taxi --max-points 2000
   $ pysad-live --source creditcard --max-points 5000

   # Your own CSV file
   $ pysad-live --source data.csv --csv-delimiter ";" --no-csv-header --detector rs_hash

``--source`` takes ``synthetic``, ``nyc_taxi``, ``arrhythmia``, ``creditcard``, or a path to a CSV file. The default is ``synthetic``. Each row of a CSV file becomes one stream point. Use ``--csv-delimiter`` for another single-character delimiter. Use ``--no-csv-header`` when the file has no header row to skip.

Detector and calibrator
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   $ pysad-live --source synthetic --detector xstream --calibrator conformal --threshold 0.99

``--detector`` picks the detector. The default is ``loda``. ``--calibrator`` turns each raw score into a calibrated value between 0 and 1. The default is ``conformal``. Use ``--threshold`` to set the alert threshold for the calibrated score. The default is 0.95.

Pass extra constructor arguments as JSON strings with ``--detector-kwargs`` and ``--calibrator-kwargs``. Use ``--detector-seed`` to set the random seed. The default seed is 42.

Display
~~~~~~~

Use ``--width``, ``--height``, ``--max-features``, ``--history``, and ``--refresh-rate`` to control the terminal dashboard. Use ``--no-labels`` to hide ground truth labels. Use ``--no-ribbon`` to hide the conformal ribbon.

Web dashboard
~~~~~~~~~~~~~

Add ``--web`` to open the web dashboard in your browser. Use ``--port`` and ``--host`` to choose where the server listens. The defaults are port 8080 and host 127.0.0.1.

.. code-block:: console

   $ pysad-live --source synthetic --web --host 0.0.0.0 --port 9090

Output
~~~~~~

Add ``--json-output`` to print one JSON line per point instead of the terminal UI. Use ``--log-file`` to log alerts to a file in JSON lines format. Use ``--max-points`` to stop after a fixed number of points.

See :doc:`/guides/live_dashboard` for the full usage and :doc:`/api/live` for the command reference.

pysad.mcp
---------

The :mod:`pysad.mcp` module exposes PySAD detectors, drift meters, and explainability through the Model Context Protocol (MCP). Install the optional extra:

.. code-block:: console

   $ pip install pysad[mcp]

Run the server:

.. code-block:: console

   $ pysad-mcp

Or run the module directly:

.. code-block:: console

   $ python -m pysad.mcp.server

The server uses stdio transport, so it needs no network port or host. Connect an MCP client such as Claude Desktop, Cursor, or VS Code. See :doc:`/guides/mcp` for the client configuration and the list of tools.

Optional PyO3 kernels
---------------------

PySAD works with pure NumPy. Install the ``pysad[kernels]`` extra to speed up the traversal hot path of some detectors:

.. code-block:: console

   $ pip install "pysad[kernels]"

HalfSpaceTrees and RobustRandomCutForest use the available tree kernels automatically. LODA and StreamKMPlusPlus use NumPy by default. Set ``use_rust=True`` to use the density kernel for LODA or the weighted Lloyd kernel for StreamKMPlusPlus.

.. code-block:: python

   from pysad.models import LODA, StreamKMPlusPlus

   model = LODA(use_rust=True, random_state=0)
   model = StreamKMPlusPlus(use_rust=True, random_state=0)

If a kernel is unavailable, the detector uses NumPy. Pickles store the option and model arrays, so they also work without the extension.

To build from a source checkout, install ``maturin`` and run these commands from the repository root. This requires a Rust toolchain.

.. code-block:: console

   $ pip install maturin
   $ maturin build --manifest-path rust/Cargo.toml --release --locked
   $ pip install rust/target/wheels/*.whl

See :doc:`/benchmarks/kernels` for the traversal timings of the optional kernels.

Next steps
----------

- :doc:`/guides/live_dashboard` shows how to watch streaming scores and alerts as points arrive.
- :doc:`/guides/mcp` explains the MCP client configuration and tools.
- :doc:`/guides/production` covers warm-up from history, per-source detectors, and latency measurement.
- :doc:`/benchmarks/kernels` times the optional kernels.
