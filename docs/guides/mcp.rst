MCP Server for PySAD
====================

PySAD includes a Model Context Protocol (MCP) server that exposes streaming anomaly
detection capabilities to AI assistants and LLM clients. The server provides tools
for running detectors, drift meters, and explainability methods.

Installation
------------

The MCP server requires the ``mcp`` optional dependency:

.. code-block:: console

   $ pip install pysad[mcp]

This installs PySAD with the ``mcp`` package (version 1.26 or newer).

Running the Server
------------------

Start the MCP server via the console script:

.. code-block:: console

   $ pysad-mcp

Or run the module directly:

.. code-block:: console

   $ python -m pysad.mcp.server

The server communicates over stdio transport, which is the standard MCP transport
for local LLM clients like Claude Desktop, Cursor, or VS Code with MCP extensions.

Configuring MCP Clients
-----------------------

To use the PySAD MCP server with an MCP client, add it to your client configuration.

**Claude Desktop** (``~/Library/Application Support/Claude/claude_desktop_config.json`` on macOS):

.. code-block:: json

   {
     "mcpServers": {
       "pysad": {
         "command": "pysad-mcp",
         "args": []
       }
     }
   }

**Cursor** (``~/.cursor/mcp.json``):

.. code-block:: json

   {
     "mcpServers": {
       "pysad": {
         "command": "pysad-mcp",
         "args": []
       }
     }
   }

**VS Code with MCP extension** (``.vscode/mcp.json``):

.. code-block:: json

   {
     "servers": {
       "pysad": {
         "command": "pysad-mcp",
         "args": []
       }
     }
   }

After configuring, restart your MCP client to connect to the server.

Available Tools
---------------

The server exposes two categories of tools:

1. **Session-based tools** (backward compatible): Manage a stateful detection
   session with ``initialize_detector``, ``initialize_calibrator``,
   ``initialize_stream``, ``score_point``, ``score_batch``, etc.

2. **Auto-discovered tools** (one per component): Direct access to every detector,
   drift meter, and statistic in PySAD with explicit parameters.

Session-Based Tools
~~~~~~~~~~~~~~~~~~~

These tools maintain a session state across calls:

- ``initialize_detector``: Initialize a streaming anomaly detector
- ``initialize_calibrator``: Initialize a probability calibrator
- ``initialize_stream``: Initialize a data stream (synthetic, NYC Taxi, Arrhythmia, Credit Card, or CSV)
- ``score_point``: Score a single point (or consume from stream)
- ``score_batch``: Score a batch of points
- ``get_status``: Get current session status
- ``reset_session``: Reset the session
- ``list_detectors``: List available detectors
- ``list_calibrators``: List available calibrators
- ``list_data_sources``: List available data sources

Auto-Discovered Detector Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every detector in ``pysad.models.__all__`` is exposed as a tool named
``pysad_<detector_name_lowercase>``. For example:

- ``pysad_loda``: LODA detector
- ``pysad_halfspacetrees``: HalfSpaceTrees detector
- ``pysad_onlineisolationforest``: OnlineIsolationForest detector
- ``pysad_rshash``: RSHash detector
- ``pysad_robustrandomcutforest``: RobustRandomCutForest detector
- And 60+ more...

Each detector tool accepts:

.. code-block:: json

   {
     "data": [[1.0, 2.0], [2.0, 3.0]],
     "fit": true,
     "num_bins": 10,
     "num_random_cuts": 100,
     "random_state": 42
   }

The ``data`` parameter is a list of feature vectors (or a single vector).
The ``fit`` parameter controls whether to fit the model (``true``) or only
score (``false``). All other parameters are detector-specific constructor arguments.

Auto-Discovered Drift Meter & Statistic Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every statistic in ``pysad.statistics.__all__`` is exposed as a tool named
``pysad_<statistic_name_lowercase>``. For example:

- ``pysad_adwinmeter``: ADWIN drift detector
- ``pysad_ddmmeter``: DDM drift detector
- ``pysad_eddmmeter``: EDDM drift detector
- ``pysad_pagehinkleymeter``: Page-Hinkley statistic
- ``pysad_kswinmeter``: KSWIN drift detector
- ``pysad_hddmameter``: HDDM_A drift detector
- ``pysad_hddmwmeter``: HDDM_W drift detector
- And more (running statistics, quantile meters, etc.)

Each statistic tool accepts:

.. code-block:: json

   {
     "data": [1.0, 1.1, 1.0, 5.0, 5.1],
     "delta": 0.002,
     "max_buckets": 5
   }

The ``data`` parameter is a list of scalar values. All other parameters are
statistic-specific constructor arguments.

Explainability Tool
~~~~~~~~~~~~~~~~~~~

- ``pysad_explain``: Get SHAP values or feature attributions for detectors
  that support ``explain_partial`` (e.g., OnlineIsolationForest, LODA,
  HalfSpaceTrees, KitNet, FeedbackHST).

Example:

.. code-block:: json

   {
     "detector_name": "OnlineIsolationForest",
     "data": [[1.0, 2.0], [2.0, 3.0], [10.0, 20.0]],
     "detector_kwargs": {"num_trees": 100, "window_size": 2048}
   }

Returns per-instance SHAP values explaining the anomaly score.

Stream Pipeline Tool
~~~~~~~~~~~~~~~~~~~~

- ``pysad_stream``: Run a complete streaming pipeline with detector + optional calibrator.

Example:

.. code-block:: json

   {
     "detector_name": "LODA",
     "data": [[1.0, 2.0], [2.0, 3.0], [10.0, 20.0]],
     "calibrator_name": "ConformalProbabilityCalibrator",
     "detector_kwargs": {"num_bins": 10, "num_random_cuts": 100},
     "threshold": 0.95
   }

Returns raw scores, calibrated scores, and alert decisions for each point.

Example Usage with an LLM Client
--------------------------------

Once the MCP server is configured, you can ask your LLM client to:

* "Run LODA on this data: [[1,2], [2,3], [100,200]]"
* "Check for concept drift in this stream using ADWIN: [1, 1.1, 1.0, 5, 5.1]"
* "Explain why OnlineIsolationForest flagged this point as anomalous"
* "Run a full anomaly detection pipeline with LODA and conformal calibration"

The LLM will automatically discover and call the appropriate PySAD tools.

Programmatic Access
-------------------

You can also use the MCP server programmatically in Python:

.. code-block:: python

   import asyncio
   from pysad.mcp.server import create_mcp_server

   async def main():
       server = create_mcp_server()
       
       # Call a detector tool directly
       result = await server.call_tool("pysad_loda", {
           "data": [[1.0, 2.0], [2.0, 3.0], [10.0, 20.0]],
           "fit": True,
           "num_bins": 10,
           "num_random_cuts": 100
       })
       print(result)
       
       # Call a drift meter
       result = await server.call_tool("pysad_adwinmeter", {
           "data": [1.0, 1.1, 1.0, 1.05, 5.0, 5.1, 5.0]
       })
       print(result)

   asyncio.run(main())

Requirements
------------

- Python 3.11+
- PySAD with ``mcp`` extra (``pip install pysad[mcp]``)
- MCP client (Claude Desktop, Cursor, VS Code, etc.)

The server uses stdio transport, so no network configuration is needed.