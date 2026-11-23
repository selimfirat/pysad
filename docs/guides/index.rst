How-to guides
=============

Choose a guide for the task you need to complete. These guides assume familiarity with the score-then-learn loop in :doc:`/getting_started`. For background, see :doc:`/concepts`. For parameters and method signatures, see :doc:`/api`.

.. container:: cards

   .. container:: card

      :doc:`stream_data`

      Stream CSV files, DataFrames, sensor values, messages and benchmark datasets.

   .. container:: card

      :doc:`dataset_inventory`

      List each pysad.utils.Data loader with its source, license, checksum and label format.

   .. container:: card

      :doc:`choose_detector`

      Find candidate detectors for your data and compare them.

   .. container:: card

      :doc:`alerts`

      Calibrate scores, choose a threshold and keep anomalies out of the model.

   .. container:: card

      :doc:`evaluate`

      Choose a metric, follow performance over time and report stable results.

   .. container:: card

      :doc:`pipelines`

      Add preprocessing, projections, postprocessing and ensembles.

   .. container:: card

      :doc:`drift_aware_composition`

      Reset a detector on drift and chain stages into one pipeline.

   .. container:: card

      :doc:`seasonal`

      Find anomalies in metrics with a daily or weekly cycle.

   .. container:: card

      :doc:`pyod`

      Run a PyOD batch detector on a stream.

   .. container:: card

      :doc:`production`

      Start from historical data, save the state, run one detector per source and control the time per point.

   .. container:: card

      :doc:`serving`

      Consume from Kafka or Redpanda, expose a FastAPI streaming endpoint, and checkpoint state to disk or Redis.

   .. container:: card

      :doc:`custom_model`

      Implement the model interface for a published streaming method and add it to PySAD.

   .. container:: card

      :doc:`mcp`

      Run PySAD detectors, drift meters, and explainability from AI assistants via MCP.

   .. container:: card

      :doc:`/examples`

      Complete scripts that run end to end.

Migration guides
----------------

.. container:: cards

   .. container:: card

      :doc:`migrate_azure`

      Replace retired Azure AI Anomaly Detector and Metrics Advisor in Python.

   .. container:: card

      :doc:`migrate_aws`

      Replace AWS Lookout for Equipment and Kinesis RANDOM_CUT_FOREST with PySAD.

   .. container:: card

      :doc:`migrate_merlion`

      Migrate anomaly detection pipelines from archived Salesforce Merlion.

   .. container:: card

      :doc:`migrate_alibi_detect`

      Move from BSL 1.1 Alibi Detect to permissive BSD 3-Clause PySAD.

   .. container:: card

      :doc:`migrate_statsforecast`

      Replace StatsForecast and Nixtla batch forecasts with PySAD streaming pipelines.

   .. container:: card

      :doc:`migrate_adtk`

      Map ADTK batch rule-based and statistical detection to PySAD streaming detection.
