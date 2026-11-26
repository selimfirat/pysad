MCP Server
==========

.. module:: pysad.mcp

The :mod:`pysad.mcp` module provides a Model Context Protocol (MCP) server that
exposes PySAD streaming anomaly detection as tools for LLM clients.

The server requires the optional ``mcp`` dependency:

.. code-block:: console

   $ pip install pysad[mcp]

Run the server with the ``pysad-mcp`` console script:

.. code-block:: console

   $ pysad-mcp

Or run the module directly:

.. code-block:: console

   $ python -m pysad.mcp.server

The server uses stdio transport, so no network configuration is needed.

Session-based tools
-------------------

These tools keep session state across calls:

- ``initialize_detector``: initialize a streaming anomaly detector
- ``initialize_calibrator``: initialize a probability calibrator
- ``initialize_stream``: initialize a data stream
- ``score_point``: score a single point
- ``score_batch``: score a batch of points
- ``get_status``: get the current session status
- ``reset_session``: reset the session
- ``list_detectors``: list available detectors
- ``list_calibrators``: list available calibrators
- ``list_data_sources``: list available data sources

Auto-discovered server tools
----------------------------

The server registers one optional tool per detector and statistic:

- ``pysad_<detector>``: run a PySAD streaming anomaly detector, for example
  ``pysad_loda`` or ``pysad_halfspacetrees``
- ``pysad_<statistic>``: run a PySAD drift meter or statistic, for example
  ``pysad_adwinmeter`` or ``pysad_ddmmeter``
- ``pysad_explain``: explain anomaly scores with SHAP or feature attribution
- ``pysad_stream``: run a full streaming pipeline with optional calibration

These tools are discovered from ``pysad.models.__all__`` and
``pysad.statistics.__all__``. Each tool lists its constructor parameters and
defaults in its metadata.

Programmatic access
-------------------

You can build and call the server directly in Python:

.. code-block:: python

   import asyncio
   from pysad.mcp.server import create_mcp_server

   async def main():
       server = create_mcp_server()
       result = await server.call_tool("pysad_loda", {
           "data": [[1.0, 2.0], [2.0, 3.0], [10.0, 20.0]],
           "fit": True,
           "num_bins": 10,
           "num_random_cuts": 100,
       })
       print(result)

   asyncio.run(main())

API reference
-------------

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :kind: classes

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :kind: functions
   :exclude: mcp_server

Server instance
~~~~~~~~~~~~~~~

The ``mcp_server`` instance is the configured MCP server returned by
:func:`create_mcp_server`. Call :func:`main` to run it over stdio transport.

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   mcp_server
