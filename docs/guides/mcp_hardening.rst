.. meta::
   :description: Harden the PySAD MCP server for production: run as a dedicated unprivileged user, restrict data sources and batch sizes, add auth and rate limiting, terminate TLS at a reverse proxy, avoid error leakage, pin versions, and monitor tool calls.

Harden the PySAD MCP server for production
==========================================

Run the PySAD MCP server safely in production: isolate the process as a dedicated user, restrict tool inputs, add authentication, rate limiting, and TLS in front of it, and monitor calls.

The PySAD MCP server exposes streaming detectors, drift meters, and data streams as tools for LLM clients. It runs over stdio transport and opens no network port. See :doc:`mcp` for installation and client configuration.

This guide assumes the optional ``mcp`` dependency is installed:

.. code-block:: console

   $ pip install pysad[mcp]

Run in a dedicated process
--------------------------

Do not run ``pysad-mcp`` as root. Create a dedicated unprivileged user that owns only the files it needs, and set memory and CPU limits with your process manager or container. The server loads detector state into memory, so cap the memory available to the process:

.. code-block:: console

   $ useradd --system --create-home pysad-mcp
   $ sudo -u pysad-mcp pysad-mcp

With systemd, run the service as that user:

.. code-block:: ini

   [Service]
   User=pysad-mcp
   Group=pysad-mcp
   ExecStart=pysad-mcp
   Restart=always
   NoNewPrivileges=true

Keep checkpoint and state files out of world-writable directories. Load pickles only from sources you trust. A pickle can run code when you load it.

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

The MCP protocol has no built-in authentication, API key, or token. Clients that connect locally (Claude Desktop, Cursor, VS Code) are bounded by the client user's permissions. Only run the server for clients that you trust.

If you expose the server to other machines or multiple users, run one server process per user or wrap it in an authentication layer in your application environment (PySAD does not install an auth package):

- Place an OAuth2 proxy or API gateway (such as Ory Oathkeeper, Kong, or a small FastAPI wrapper) in front of an HTTP wrapper that checks a bearer token or session cookie before forwarding authorized calls to ``pysad-mcp`` over stdio.
- Do not pass user credentials into detector tool arguments; treat tool arguments as untrusted input.

Rate limiting
-------------

The server has no built-in rate limiter, so a client can call detector tools as fast as the process allows. Limit calls in your application environment with a token bucket wrapper or a reverse proxy sidecar such as nginx ``limit_req``:

.. code-block:: python

   import asyncio
   import time

   from pysad.mcp.server import create_mcp_server


   class RateLimiter:
       """Allow at most ``rate`` calls per second."""

       def __init__(self, rate: float):
           self.rate = rate
           self.tokens = rate
           self.updated = time.monotonic()
           self.lock = asyncio.Lock()

       async def acquire(self) -> None:
           async with self.lock:
               now = time.monotonic()
               self.tokens += (now - self.updated) * self.rate
               self.updated = now
               if self.tokens > self.rate:
                   self.tokens = self.rate
               while self.tokens < 1.0:
                   await asyncio.sleep((1.0 - self.tokens) / self.rate)
                   now = time.monotonic()
                   self.tokens += (now - self.updated) * self.rate
                   self.updated = now
               self.tokens -= 1.0


   limiter = RateLimiter(rate=10.0)


   async def call_tool_limited(server, name, arguments):
       await limiter.acquire()
       return await server.call_tool(name, arguments)


   async def main():
       server = create_mcp_server()
       await call_tool_limited(server, "list_detectors", {})


   asyncio.run(main())

TLS
---

Because ``pysad-mcp`` carries no network traffic, TLS does not apply to stdio directly. If you expose PySAD scores over HTTP or WebSocket instead, terminate TLS at a reverse proxy in front of a localhost backend (see :doc:`serving`):

.. code-block:: nginx

   server {
       listen 443 ssl;
       server_name pysad.example.com;

       ssl_certificate /etc/letsencrypt/live/pysad.example.com/fullchain.pem;
       ssl_certificate_key /etc/letsencrypt/live/pysad.example.com/privkey.pem;

       location /score {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }

Reload nginx (``nginx -s reload``), keep the certificate current, and bind the backend to localhost so clients cannot bypass TLS.

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

- Run the server as a dedicated non-root user with memory and CPU limits, keeping checkpoints out of world-writable directories.
- Use stdio only, or place auth, rate limiting, and TLS termination in front of any HTTP wrapper.
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
