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

This installs PySAD with the ``mcp`` package (version 2.x).

Agent skills and plugin
-----------------------

``pysad.agent`` bundles two skills and a Claude Code plugin. The skills cover
stream scoring and feature explanations through the existing MCP tools.
They ship with the base package and add no runtime dependencies.
The server still requires ``pysad[mcp]``.

To locate the plugin in your Python environment, run:

.. code-block:: console

   $ python -m pysad.agent

If you use Claude Code, load the plugin with:

.. code-block:: console

   $ claude --plugin-dir "$(python -m pysad.agent)"

The plugin starts ``pysad-mcp`` from your ``PATH``. Activate the Python environment
with ``pysad[mcp]`` before starting the client. The plugin includes the server
configuration, so you do not need a second MCP entry for this setup.
See the `plugin manifest reference <https://code.claude.com/docs/en/plugins-reference>`_
for the client format.

If your client supports Agent Skills, locate the skill folders with:

.. code-block:: console

   $ python -m pysad.agent --skills

Copy ``pysad-detect`` and ``pysad-explain`` from that directory into your client's
skills directory. Configure the MCP server as described below.
The skills use the tool schemas from the connected server.
Each detector tool call starts a fresh model. Separate calls do not continue
training from earlier calls.

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

PySAD ships a root-level ``.mcp.json`` that declares ``pysad`` (``command: pysad-mcp``). Clients that scan the workspace root, such as Cursor, discover it automatically. For manual setup (Claude Desktop at ``~/Library/Application Support/Claude/claude_desktop_config.json``, Cursor at ``~/.cursor/mcp.json``, or VS Code in ``.vscode/mcp.json`` under ``"servers"``), configure:

.. code-block:: json

   {
     "mcpServers": {
       "pysad": {
         "command": "pysad-mcp",
         "args": []
       }
     }
   }

Restart your MCP client after updating the configuration.

Available Tools
---------------

The server exposes **session-based tools** (stateful per session: ``initialize_detector``, ``initialize_calibrator``, ``initialize_stream``, ``score_point``, ``score_batch``, ``get_status``, ``reset_session``, ``list_detectors``, ``list_calibrators``, ``list_data_sources``) and **auto-discovered tools** (stateless per call, generated from ``pysad.models.__all__`` and ``pysad.statistics.__all__``).

Auto-Discovered Detector and Statistic Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every detector in ``pysad.models.__all__`` is registered as ``pysad_<detector_name_lowercase>`` (for example ``pysad_loda``, ``pysad_halfspacetrees``, ``pysad_onlineisolationforest``, ``pysad_rshash``, ``pysad_robustrandomcutforest``). Pass ``data`` (feature vector or list of vectors), optional ``fit`` (default ``true``), and detector parameters.

Every statistic or drift meter in ``pysad.statistics.__all__`` is registered as ``pysad_<statistic_name_lowercase>`` (for example ``pysad_adwinmeter``, ``pysad_ddmmeter``, ``pysad_eddmmeter``, ``pysad_pagehinkleymeter``, ``pysad_kswinmeter``). Pass ``data`` (list of scalar floats) and constructor arguments.

Explainability and Stream Pipeline Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``pysad_explain``: Computes per-instance attributions for detectors with ``explain_partial`` (such as ``OnlineIsolationForest``, ``LODA``, ``HalfSpaceTrees``, ``KitNet``, ``FeedbackHST``).
- ``pysad_stream``: Runs a full pipeline pairing a detector with an optional calibrator (for example ``ConformalProbabilityCalibrator``) and threshold (default ``0.95``), returning raw scores, calibrated scores, and alerts.

Programmatic Access and Monitoring
----------------------------------

Use :func:`~pysad.mcp.server.create_mcp_server` in Python to invoke tools or inspect ``get_status`` for health checks and session progress:

.. code-block:: python

   import asyncio
   from pysad.mcp.server import create_mcp_server

   async def main():
       server = create_mcp_server()
       _, status = await server.call_tool("get_status", {})
       print("Status:", status)
       result = await server.call_tool(
           "pysad_loda",
           {"data": [[1.0, 2.0], [2.0, 3.0], [10.0, 20.0]], "fit": True, "num_bins": 10},
       )
       print(result)

   asyncio.run(main())

The server logs lifecycle and error events via ``ctx.info`` and ``ctx.error``. Wrap ``call_tool`` in your application process to increment Prometheus counters (see :doc:`serving` for the HTTP exporter pattern). Each standalone ``call_tool`` call uses a fresh context; persistent sessions require a single client connection or external pickle snapshots.

Hardening for Production
------------------------

The MCP protocol carries no built-in authentication or rate limiting, and ``pysad-mcp`` communicates only over local stdio:

- **Dedicated unprivileged process**: Run ``pysad-mcp`` as a non-root service user with container/systemd memory and CPU limits (``NoNewPrivileges=true``), keeping checkpoints out of world-writable directories.
- **Keep stdio local or wrap safely**: Never expose raw stdio over a network socket. If exposing HTTP access, place OAuth2/bearer auth, nginx/token-bucket rate limiting, and TLS termination in front of a FastAPI backend (:doc:`serving`).
- **Restrict inputs and batches**: In production wrappers around ``create_mcp_server``, allow only trusted ``initialize_stream`` sources (reject arbitrary ``.csv`` paths), enforce maximum batch sizes on ``score_batch``, and sanitize internal exception text returned to clients.
- **Pin versions and reset state**: Pin both ``pysad`` and ``mcp`` versions, call ``reset_session`` between unrelated jobs, and load pickles only from trusted sources.
