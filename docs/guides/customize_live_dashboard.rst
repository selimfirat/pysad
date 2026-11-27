.. meta::
   :description: Customize the pysad-live dashboard: choose what it shows, tune terminal and web display size and refresh rate, and change the dashboard colors.

Customize the pysad-live dashboard
==================================

The ``pysad-live`` command shows one dashboard for a running stream. You can choose what it shows, how large it is, how often it updates, and which colors it uses.

This guide covers the customization that works today. For the full list of command options, see :doc:`/guides/configuration`. For how to open and use the dashboard, see :doc:`/guides/live_dashboard`.

Choose what the dashboard shows
-------------------------------

The terminal dashboard shows features, the score, the calibrated score, the conformal ribbon, and recent alerts. Use these options to change what appears:

- ``--max-features`` sets how many feature cards to show (default: 4).
- ``--history`` sets how many points the sparklines and ribbon remember (default: 100).
- ``--no-labels`` hides ground truth labels.
- ``--no-ribbon`` hides the conformal ribbon.

.. code-block:: console

   $ pysad-live --source synthetic --web --max-features 2 --no-labels --no-ribbon

Control timing and size
-----------------------

- ``--width`` and ``--height`` set the terminal size (defaults: 80 by 24).
- ``--refresh-rate`` sets the seconds between points (default: 0.1).
- ``--max-points`` stops the stream after a fixed number of points.

.. code-block:: console

   $ pysad-live --source nyc_taxi --width 120 --refresh-rate 0.05 --max-points 2000

The web dashboard ignores ``--width`` and ``--height``. It fills the browser window. Use ``--port`` and ``--host`` to choose where it listens:

.. code-block:: console

   $ pysad-live --source synthetic --web --host 0.0.0.0 --port 9090

Change the output
-----------------

- ``--json-output`` prints one JSON line per point instead of the terminal UI.
- ``--log-file`` appends one JSON line per alert to a file.

See :doc:`/guides/production` for running pysad-live with JSON output in production.

Choose the detector, calibrator and threshold
---------------------------------------------

Use ``--detector``, ``--calibrator``, ``--threshold``, ``--detector-kwargs`` and ``--calibrator-kwargs`` to score the stream your way. The default detector is LODA, the default calibrator is conformal, and the default alert threshold is 0.95.

.. code-block:: console

   $ pysad-live --source data.csv --detector half_space_trees --calibrator spot --threshold 0.99

See :doc:`/guides/configuration` for all data source, detector, calibrator, display, web dashboard, and output options.

Change the dashboard colors
---------------------------

The web dashboard has one fixed dark theme. The colors live in the ``DASHBOARD_HTML`` string in ``pysad/live/server.py``. There is no ``--theme`` option and no light theme today.

These are the main colors:

- Page background: ``#0d1b22``
- Panel background: ``#142833``
- Score and normal values: ``#00d4aa`` (teal)
- Alert values: ``#ff4444`` (red)
- Threshold line and warnings: ``#ffaa00`` (amber)
- Muted text and borders: ``#7a919c`` and ``#1e3a4a``

Edit the color strings in the ``<style>`` block of ``DASHBOARD_HTML`` to use your own colors. Run from the source checkout or reinstall PySAD for the change to take effect.

The terminal dashboard uses ANSI color constants at the top of ``pysad/live/terminal.py``:

- ``SCORE_COLOR = GREEN``
- ``ALERT_COLOR = RED``
- ``THRESHOLD_COLOR = YELLOW``
- ``RIBBON_COLOR = BLUE``
- ``FEATURE_COLORS`` holds one color per feature card.

Edit these constants to change the terminal colors.

Next steps
----------

- :doc:`/guides/live_dashboard` shows how to open the dashboard and watch scores and alerts.
- :doc:`/guides/configuration` lists every ``pysad-live`` option.
- :doc:`/api/live` documents the ``pysad-live`` command.
- :doc:`/guides/production` covers running pysad-live in production.
