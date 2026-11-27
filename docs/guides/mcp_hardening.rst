.. meta::
   :description: Harden the PySAD MCP server for production: run in a dedicated process, restrict data sources and batch sizes, avoid error leakage, pin versions, and monitor tool calls.

Harden the PySAD MCP server for production
==========================================

Run the PySAD MCP server safely in production: isolate the process, restrict tool inputs, and monitor calls.

The PySAD MCP server exposes streaming detectors, drift meters, and data streams as tools for LLM clients. It runs over stdio transport and opens no network port. See :doc:`mcp` for installation and client configuration.

This guide assumes the optional ``mcp`` dependency is installed:

.. code-block:: console

   $ pip install pysad[mcp]

Run in a dedicated process
--------------------------

Run ``pysad-mcp`` in its own process as a non-root user. Set memory and CPU limits with your process manager or container. The server loads detector state into memory, so cap the memory available to the process.

Stdio transport means the server serves the local client that launched it. Do not expose the stdio stream over a network socket. If you need HTTP access, use the FastAPI recipes in :doc:`serving` instead of the MCP server.

Restrict data sources
---------------------

The ``initialize_stream`` tool reads any file whose path ends in ``.csv``. A client can point it at any file that the process user can read. In production, do one of the following:

- Allow only the built-in sources (``synthetic``, ``nyc_taxi``, ``arrhythmia``, ``creditcard``) and reject paths that end in ``.csv``, or
- Wrap ``create_mcp_server`` and reject ``initialize_stream`` calls with an unknown source.

Never point the server at a directory that contains secrets or source code.

Limit batch sizes
-----------------

The ``score_batch`` tool accepts an unbounded list of points. A large batch can block the server and delay other calls. Wrap ``score_batch`` and reject batches above a fixed size, for example 1,000 points. Split larger inputs into several calls.

Also limit the number of features per point. A detector fixes the feature count at the first point. Reject points whose length differs from the first point.

Handle errors safely
--------------------

Tool failures return dictionaries with ``"success": False`` and an error string. The error string can contain details from the underlying detector. In production, wrap ``call_tool`` and replace client-facing error messages with generic text. Log the full error on the server side only.

Do not return detector internals, file paths, or stack traces to the client.

Authenticate the client
-----------------------

The MCP protocol has no built-in authentication. The server trusts whoever connects over stdio. Only run the server for clients that you trust. If multiple users share the machine, run one server process per user or add an access control layer in front of the stdio connection.

Manage session state
--------------------

Session state lives on the MCP context and does not survive a restart. The detector, calibrator, and stream are lost when the process stops. Save detector and calibrator state with ``pickle`` outside the server and restore it on startup. See :doc:`production` for pickling and version pinning.

Reset the session between unrelated jobs with ``reset_session`` so scores do not carry over.

Pin versions
------------

Pin PySAD and the ``mcp`` package in your requirements:

.. code-block:: text

   pysad==0.8.0
   mcp==2.3.0

Server behavior depends on the PySAD version. Detector scores can change between releases.

Monitor tool calls
------------------

Wrap ``create_mcp_server`` and count calls per tool with a Prometheus counter. See the ``pysad.mcp`` recipe in :doc:`serving`. Alert on a sudden rise in calls or errors.

Production checklist
--------------------

- Run the server as a non-root user with memory and CPU limits.
- Use stdio only; do not expose a network port.
- Restrict ``initialize_stream`` to trusted sources.
- Limit ``score_batch`` to a fixed number of points per call.
- Replace client-facing error messages with generic text.
- Pin ``pysad`` and ``mcp`` versions.
- Save and restore detector state across restarts.
- Monitor tool calls and error rates.

Next steps
----------

- :doc:`mcp` describes the server tools and client configuration.
- :doc:`production` describes warm-up from history, per-source detectors, and latency measurement.
- :doc:`serving` describes Prometheus metrics and checkpoints.
