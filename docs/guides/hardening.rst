.. meta::
   :description: Harden the pysad.mcp server in production: run as a dedicated user, add authentication and rate limiting in front of the server, and terminate TLS at a reverse proxy.

Harden the pysad.mcp server in production
=========================================

Run the PySAD MCP server with authentication, rate limiting, and TLS in front of it.

The ``pysad-mcp`` server speaks MCP over stdio. It opens no network port, and PySAD does not build authentication, rate limiting, or TLS into the server. Harden it with your infrastructure: run it as a dedicated unprivileged user, put an authentication layer and a rate limiter in front of it, and terminate TLS at a reverse proxy if clients reach it over HTTP.

Install the optional mcp extra:

.. code-block:: console

   $ pip install pysad[mcp]

Run as a dedicated user
-----------------------

Do not run the server as root. Create a service user that owns only the files it needs:

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

Authentication
--------------

The stdio server has no login, API key, or token. Clients that connect through a local MCP client (Claude Desktop, Cursor, VS Code) are already bounded by the client user's permissions.

If you expose the server to other machines or users, wrap it in an authentication layer that runs in your application environment, not in PySAD:

- Place an OAuth2 proxy or an API gateway in front of an HTTP wrapper around the server.
- The wrapper checks a bearer token or session cookie, then forwards authorized calls to ``pysad-mcp`` over stdio.
- Do not pass user credentials into detector tool arguments. Treat tool arguments as untrusted input.

PySAD does not install an auth package. Choose one that fits your infrastructure, for example Ory Oathkeeper, Kong, or a small FastAPI wrapper.

Rate limiting
-------------

The server has no built-in rate limiter. A client can call detector tools as fast as the process allows. Limit calls in your application environment.

Token bucket wrapper
~~~~~~~~~~~~~~~~~~~~

Wrap ``call_tool`` with a token bucket that allows a fixed number of calls per second:

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

   limiter = RateLimiter(rate=10.0)  # 10 calls per second

   async def call_tool_limited(server, name, arguments):
       await limiter.acquire()
       return await server.call_tool(name, arguments)


   async def main():
       server = create_mcp_server()
       await call_tool_limited(server, "list_detectors", {})

   asyncio.run(main())

Adjust ``rate`` to your traffic. A reverse proxy or sidecar, for example nginx ``limit_req``, can enforce the same limit across several server processes.

TLS
---

The stdio server carries no network traffic, so TLS does not apply to it directly. If you expose PySAD scores over HTTP or WebSocket instead, terminate TLS at a reverse proxy in front of your application.

nginx example
~~~~~~~~~~~~~

This nginx configuration terminates TLS and forwards to a local FastAPI serving endpoint (see :doc:`serving`):

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

Reload nginx:

.. code-block:: console

   $ nginx -s reload

Keep the certificate current and restrict the backend to localhost so clients cannot bypass TLS.

Next steps
----------

- :doc:`mcp` shows the client configuration and the list of MCP tools.
- :doc:`serving` shows FastAPI and WebSocket endpoints you can expose behind TLS.
- :doc:`production` shows warm-up, state saving, and per-source detectors.
