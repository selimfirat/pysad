.. meta::
   :description: Monitor and observe the pysad.mcp MCP server: check session status with get_status, read log messages, count tool calls, and run health checks.

Monitor the pysad.mcp server
============================

This guide shows how to observe a running ``pysad.mcp`` server. Use it to check session state, read log messages, count tool calls, and run health checks.

The server runs over stdio and keeps one session per MCP client connection. Session-based tools such as ``initialize_detector`` and ``score_point`` keep detector and calibrator state across calls. See :doc:`mcp` for installation, client configuration, and the full list of tools.

Check the session status
------------------------

The ``get_status`` tool reports the current state of a session:

.. code-block:: python

   import asyncio
   from pysad.mcp.server import create_mcp_server

   async def main():
       server = create_mcp_server()
       _, info = await server.call_tool("get_status", {})
       print(info)

   asyncio.run(main())

It returns whether a detector is initialized, the detector and calibrator names, the number of scored points, the alert threshold, and how many stream points remain:

- ``initialized``: True after ``initialize_detector`` ran.
- ``detector``: the detector class name, for example ``LODA``.
- ``calibrator``: the calibrator class name.
- ``step``: points scored in this session.
- ``stream_length``: total points in the initialized stream.
- ``stream_position``: the same as ``step`` while scoring from the stream.
- ``has_stream``: True after ``initialize_stream`` ran.

Call ``get_status`` regularly to detect a stalled session, for example when ``step`` stops increasing while the stream still has points.

Read the server log
-------------------

The server logs each step through the MCP context:

- ``ctx.info``: initialization, scores, batch results, resets.
- ``ctx.error``: failed initialization or scoring.

An MCP client such as Claude Desktop or Cursor shows these log messages in its user interface. Programmatically, pass a context that records them. Standalone ``call_tool`` calls create a fresh context, so log messages do not carry over between calls.

Count tool calls
----------------

Wrap ``call_tool`` in your own process to count how often each tool is used. This helps you spot a loop in an LLM client:

.. code-block:: python

   import asyncio

   from prometheus_client import Counter, start_http_server

   from pysad.mcp.server import create_mcp_server

   start_http_server(port=8001, addr="127.0.0.1")
   calls = Counter("pysad_mcp_calls_total", "MCP tool calls", ["tool"])

   async def call_tool_counted(server, name, arguments):
       calls.labels(tool=name).inc()
       return await server.call_tool(name, arguments)

   async def main():
       server = create_mcp_server()
       _, info = await call_tool_counted(server, "get_status", {})
       print(info)

   asyncio.run(main())

Prometheus scrapes ``http://127.0.0.1:8001/metrics``. The ``prometheus_client`` package belongs in your application environment, not in PySAD dependencies. See :doc:`serving` for the full Prometheus exporter recipe.

Health check
------------

Use ``get_status`` as a health check. The server is healthy when the call returns ``"success": true``. If it returns an error or the call times out, restart the server process:

.. code-block:: console

   $ pysad-mcp

The server holds no persistent state between client sessions. Restarting it drops in-progress sessions. It does not lose detector pickles, because the server does not save them.

Next steps
----------

- :doc:`mcp` explains the client configuration and every tool.
- :doc:`production` covers warm-up, saved state, and latency per point.
- :doc:`serving` shows how to wrap pysad-live and pysad.mcp with Prometheus metrics.
