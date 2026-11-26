Live CLI
========

.. module:: pysad.live

This module provides the live streaming anomaly detection CLI and dashboard.

Read a CSV file
---------------

The ``--source`` option takes a path to a CSV file. Each row becomes one stream point. PySAD reads the file with :class:`~pysad.utils.CSVStreamer`.

By default, PySAD treats the first row as a header and skips it. It assumes a comma delimiter.

Use ``--csv-delimiter`` to set another single-character field delimiter. Use ``--no-csv-header`` when the file has no header row to skip. The default delimiter is ``","``. The default is to skip a header row.

Examples:

.. code-block:: console

   # Comma-delimited CSV with a header row (default)
   pysad-live --source data.csv --detector loda

   # Semicolon-delimited CSV with no header row
   pysad-live --source data.csv --csv-delimiter ";" --no-csv-header --detector rs_hash

   # Tab-delimited CSV with a header row
   pysad-live --source data.csv --csv-delimiter $'\t' --detector loda

Open the web dashboard
----------------------

Add ``--web`` to open the web dashboard in your browser. The dashboard streams scores, conformal ribbons, and alerts with Server-Sent Events. It uses the Python standard library only, so no extra package is needed.

.. code-block:: console

   # Open the dashboard on the default port 8080
   pysad-live --source synthetic --detector loda --calibrator conformal --web

   # Choose another port and host
   pysad-live --source nyc_taxi --detector half_space_trees --calibrator spot --web --port 9090 --host 0.0.0.0

Open ``http://127.0.0.1:8080`` in your browser. See :doc:`/guides/live_dashboard` for the full usage, including data sources, detectors, calibrators, and what the dashboard shows.

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :kind: functions

   main
