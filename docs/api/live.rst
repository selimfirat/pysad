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

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :kind: functions

   main
