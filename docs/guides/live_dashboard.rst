.. meta::
   :description: Open the pysad-live web dashboard and watch streaming scores, conformal ribbons, and alerts as points arrive.

View the live dashboard
=======================

The ``pysad-live`` command starts a terminal dashboard by default. Add ``--web`` to open a web dashboard in your browser instead. The dashboard shows each feature, the anomaly score, the calibrated score, a conformal ribbon, and recent alerts. It updates as points arrive.

The web server uses the Python standard library only. It pushes updates to the browser with Server-Sent Events. No extra package is needed.

Run the dashboard
-----------------

This command streams the built-in synthetic data, scores each point with LODA, calibrates the score, and opens the dashboard on port 8080:

.. code-block:: console

   $ pysad-live --source synthetic --detector loda --calibrator conformal --web --port 8080

Open ``http://127.0.0.1:8080`` in your browser. The dashboard stays open until you stop the command with Ctrl+C.

Choose a data source
--------------------

The default source is ``synthetic``. Use ``--source`` for another built-in stream or for a CSV file:

.. code-block:: console

   # Built-in NYC taxi stream
   $ pysad-live --source nyc_taxi --detector half_space_trees --calibrator spot --web

   # Your own CSV file (comma delimiter, header row by default)
   $ pysad-live --source data.csv --detector loda --calibrator conformal --web

   # Semicolon-delimited CSV with no header row
   $ pysad-live --source data.csv --csv-delimiter ";" --no-csv-header --detector rs_hash --web

Built-in sources are synthetic, nyc_taxi, arrhythmia, and creditcard. For a CSV file, each row becomes one stream point. Use ``--csv-delimiter`` for another single-character delimiter. Use ``--no-csv-header`` when the file has no header row to skip.

Choose a detector and calibrator
--------------------------------

Use ``--detector`` and ``--calibrator`` to pick the stages. The detector scores each point before it learns from it. The calibrator turns each raw score into a value between 0 and 1.

.. code-block:: console

   $ pysad-live --source synthetic --detector xstream --calibrator conformal --threshold 0.99 --web

Common detectors are loda, half_space_trees, rs_hash, xstream, median_absolute_deviation, robust_random_cut_forest, knncad, smart_sifter, seasonal_hybrid_esd, and local_outlier_probability. Common calibrators are conformal, spot, conformal_tsfm, gaussian_tail, anomaly_likelihood, and group_conditional_conformal.

Use ``--threshold`` to set the alert threshold for the calibrated score. The default is 0.95.

Control the dashboard
---------------------

Use ``--port`` and ``--host`` to change where the server listens. The default host is 127.0.0.1 and the default port is 8080.

.. code-block:: console

   $ pysad-live --source synthetic --web --host 0.0.0.0 --port 9090

Use ``--max-points`` to stop after a fixed number of points. Use ``--refresh-rate`` to set the seconds between points. The default refresh rate is 0.1 seconds.

.. code-block:: console

   $ pysad-live --source nyc_taxi --max-points 2000 --refresh-rate 0.05 --web

What the dashboard shows
------------------------

The dashboard shows one card per feature, the current anomaly score, the calibrated score, and a ribbon of conformal bounds. A red point or bar marks an alert when the calibrated score is above the threshold. The recent alerts panel lists each alert with its step and score.

The stream worker behind the dashboard runs this loop for each point:

.. code-block:: python

   score = detector.fit_score_partial(x)
   calibrated = calibrator.fit_transform_partial(score)
   lower, upper = compute_conformal_bounds(calibrator, score)
   alert = calibrated > args.threshold

``fit_score_partial`` scores the point and then learns from it. The calibrated value is the fraction of recent scores that the new score exceeds. ``compute_conformal_bounds`` returns the lower and upper bound of the conformal ribbon around the score.

Stop the dashboard
------------------

Press Ctrl+C to stop the server and the stream worker. The command exits cleanly.

Next steps
----------

- :doc:`alerts` explains thresholds and probability calibration.
- :doc:`../api/live` documents the ``pysad-live`` command and its options.
