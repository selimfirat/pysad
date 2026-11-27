.. meta::
   :description: Secure the pysad-live dashboard and run it for multiple tenants. Bind to localhost, add a reverse proxy with authentication, and run one isolated process per tenant.

Run pysad-live with authentication and multiple tenants
=======================================================

The ``pysad-live`` command is a single-user dashboard. It has no built-in login, API token, or tenant isolation. It binds to ``127.0.0.1`` by default, so only the local machine can reach it. This guide shows how to add authentication and how to run one isolated dashboard per tenant.

If you expose the dashboard to other machines, put a reverse proxy in front of it. The raw server has no authentication, so do not expose it directly to the internet.

Bind to localhost only
----------------------

The default host is ``127.0.0.1``. Do not change it to ``0.0.0.0`` unless you add authentication in front:

.. code-block:: console

   $ pysad-live --source data.csv --detector loda --calibrator conformal --web --port 8080

Open ``http://127.0.0.1:8080`` on the same machine. The dashboard and its Server-Sent Events endpoint are not authenticated.

Add authentication with a reverse proxy
----------------------------------------

Run ``pysad-live`` on localhost and put a reverse proxy such as nginx in front of it. The proxy terminates TLS and checks credentials before it forwards a request.

If you use HTTP basic auth, install the ``apache2-utils`` package to create the password file:

.. code-block:: console

   $ sudo apt-get install apache2-utils
   $ sudo htpasswd -c /etc/nginx/.pysad_live_users alice

Add a server block that forwards ``/`` and ``/events`` to the local pysad-live port:

.. code-block:: nginx

   server {
       listen 443 ssl;
       server_name _;

       ssl_certificate /etc/letsencrypt/live/your-domain/fullchain.pem;
       ssl_certificate_key /etc/letsencrypt/live/your-domain/privkey.pem;

       location / {
           auth_basic "pysad-live";
           auth_basic_user_file /etc/nginx/.pysad_live_users;
           proxy_pass http://127.0.0.1:8080;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }

       location /events {
           auth_basic "pysad-live";
           auth_basic_user_file /etc/nginx/.pysad_live_users;
           proxy_pass http://127.0.0.1:8080;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_cache off;
           proxy_buffering off;
           proxy_read_timeout 86400s;
       }
   }

The proxy forwards the dashboard and the Server-Sent Events stream. Only users in the password file can connect. Reload nginx after you edit the file:

.. code-block:: console

   $ sudo nginx -t
   $ sudo systemctl reload nginx

If you prefer OAuth or single sign-on, use an authentication proxy such as OAuth2 Proxy or Cloudflare Access in front of ``pysad-live`` instead of basic auth.

Wrap pysad-live in your own server for API tokens
--------------------------------------------------

If you need API tokens, do not add them to ``pysad-live``. Run ``pysad-live`` with ``--json-output`` on localhost and wrap it in your own FastAPI or Flask application. Your wrapper checks the token, starts one ``pysad-live`` process per tenant, and forwards the JSON lines to the authorized client.

Run one dashboard per tenant
----------------------------

``pysad-live`` holds one detector, one calibrator, and one history in a single process. To isolate tenants, run one ``pysad-live`` process per tenant. Give each process its own port, data source, detector, calibrator, and log file:

.. code-block:: console

   # Tenant A
   $ pysad-live --source tenant_a.csv --detector loda --calibrator conformal --web --port 8101 --host 127.0.0.1 --log-file logs/tenant_a.jsonl

   # Tenant B
   $ pysad-live --source tenant_b.csv --detector half_space_trees --calibrator spot --web --port 8102 --host 127.0.0.1 --log-file logs/tenant_b.jsonl

Each process learns from its own data. Tenant A never sees Tenant B's scores, state, or alerts. Put a reverse proxy with separate credentials in front of each port if you expose them.

Use a process manager such as systemd or supervisor to restart each process if it stops. Save the port and the data source for each tenant in a config file so you can start them all together.

Isolate state in one process
----------------------------

You can also run one Python process and keep a separate detector and calibrator per tenant. The public API gives each tenant its own state:

.. code-block:: python

   from pysad.models import LODA
   from pysad.transform.probability_calibration import ConformalProbabilityCalibrator

   tenants = {}

   def get_tenant(tenant_id):
       if tenant_id not in tenants:
           tenants[tenant_id] = {
               "model": LODA(random_state=0),
               "calibrator": ConformalProbabilityCalibrator(window_size=1000),
           }
       return tenants[tenant_id]

   for tenant_id, x in stream_with_tenant_id():
       state = get_tenant(tenant_id)
       score = state["model"].fit_score_partial(x)
       probability = state["calibrator"].fit_transform_partial(score)

Each tenant has its own model and calibrator. State does not leak between tenants. Save the ``tenants`` dictionary with ``pickle`` to resume each tenant from its own state.

Next steps
----------

- :doc:`live_dashboard` shows all ``pysad-live`` options and what the dashboard displays.
- :doc:`production` covers warm-up from history, per-source detectors, and latency measurement.
- :doc:`serving` shows HTTP endpoints and checkpoints for production serving.
- :doc:`configuration` configures pysad-live, pysad.mcp, and the optional PyO3 kernels.
