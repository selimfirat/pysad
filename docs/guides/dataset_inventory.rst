Dataset inventory
=================

:class:`~pysad.utils.Data` downloads labeled datasets, checks every download against a SHA-256 checksum, and caches the files in ``~/.cache/pysad`` (or in ``PYSAD_DATA_DIR``). This page lists each loader, its source, its license, its checksum and its label format.

A loader is a collection returned by :meth:`~pysad.utils.Data.datasets`, for example ``"odds"`` or ``"nab"``. Call ``Data.datasets("odds")`` to list the file names of a loader. Call :meth:`~pysad.utils.Data.metadata` with one file name, for example ``Data.metadata("smap/P-1.npy")``, to get its source URL, license and SHA-256 checksum from the API.

.. list-table:: Dataset inventory: source, license, checksum and label format of each loader
   :header-rows: 1
   :widths: 14 26 18 22 20

   * - Loader
     - Source
     - License
     - Checksum
     - Label format
   * - **odds**
     - 17 multivariate ODDS datasets (``.mat``)
     - PyOD mirror of the ODDS library at a fixed commit; the original source is the ODDS library of Stony Brook University. License varies by dataset.
     - Per-file SHA-256 (17 files). Every download is verified.
     - Label 1 marks the anomaly class. For ``vertebral.mat`` the labels are flipped, so the unhealthy patients are the anomalies.
   * - **nab**
     - 58 univariate NAB series
     - numenta/NAB at commit ea702d75. MIT. Earlier commits of NAB are under AGPL-3.0; PySAD uses the MIT commit.
     - Per-file SHA-256 (58 series) plus ``combined_windows.json``. Every download is verified.
     - Label 1 marks the points inside an anomaly window from ``combined_windows.json``. All other points are 0.
   * - **ucr**
     - 250 univariate UCR Time Series Anomaly Archive series
     - Authors' site, UCR_TimeSeriesAnomalyDatasets2021.zip. No public redistribution license. The archive is downloaded from the authors' site and is not copied by PySAD.
     - Archive SHA-256 ac4b991c701e620ae9cc5ebd57ae45593a36cc9c0b6ed5e3c4b7e466cf4783d4. Verified on download.
     - Label 1 marks the anomaly range given by the last two numbers of the file name (counted from 1, inclusive). All other points are 0.
   * - **kddcup99**
     - KDD Cup 99 http stream of logged-in users
     - OpenML dataset 1113 (KDDCup99.arff). OpenML terms apply.
     - SHA-256 29d8e2f3d818eebe2d30f931f910c54e72a92d62ae814b6d95e8d5d1441b425e. Verified on download.
     - Label 1 marks attacks (every connection that is not normal). Label 0 marks normal connections.
   * - **smd**
     - 28 Server Machine Dataset machines (test halves)
     - NetManAIOps/OmniAnomaly, ServerMachineDataset, commit 7fb0e0ac. MIT (OmniAnomaly).
     - Per-file SHA-256 (test file and labels, 28 machines). Every download is verified.
     - Label 1 marks anomalous minutes from the matching test_label file. All other points are 0.
   * - **smap**
     - 54 SMAP satellite telemetry channels (test halves)
     - TranAD copy of the Telemanom SMAP_MSL test data, commit 7ffb98d0. BSD-3-Clause (TranAD). The Telemanom data is BSD-style.
     - Per-file SHA-256 (54 channels). Every download is verified.
     - Label 1 marks the points inside a Telemanom anomaly range. All other points are 0.
   * - **msl**
     - 27 MSL Curiosity rover telemetry channels (test halves)
     - TranAD copy of the Telemanom SMAP_MSL test data, commit 7ffb98d0. BSD-3-Clause (TranAD). The Telemanom data is BSD-style.
     - Per-file SHA-256 (27 channels). Every download is verified.
     - Label 1 marks the points inside a Telemanom anomaly range. All other points are 0.
   * - **insects**
     - 4 INSECTS streams (abrupt, incremental, incremental_gradual, incremental_reoccurring)
     - ARCUS paper on Google Drive. CC BY 4.0.
     - Per-file SHA-256 (4 streams). Every download is verified.
     - Label 1 marks the southern house mosquito, which is the anomaly class. All other points are 0.
   * - **creditcard**
     - OpenML 1597 credit-card fraud transactions (creditcard.arff)
     - OpenML dataset 1597, version 1. ODbL 1.0 for the database and DbCL 1.0 for its contents. OpenML metadata lists the license as Public.
     - SHA-256 fdaf12730dc1fc426f318b71349f24f5c5fd00aa1152940be7e7509ae3d89d2a. Verified on download.
     - Label 1 marks fraud and label 0 marks all other transactions. The Class column is returned separately.
   * - **tsb-drift**
     - 75 TSB-drift distribution-drift series
     - thedatum.org TSB-AD-M.zip (Parrino et al., KDD 2026). CC BY 4.0 for TSB-drift. Apache 2.0 for the TSB-AD curation. Underlying series keep their original licenses.
     - Archive SHA-256 7de86ac27f30eeb48d833bb061055670e3f3de07defd995cf2bd5db10ccc9a0d. Verified on download.
     - Label 1 marks an anomaly and label 0 marks a normal point. The label is the last column.
   * - **yahoo**
     - 67 Yahoo S5 A1Benchmark real production-traffic series
     - Public GitHub mirror at commit 79453c49. Yahoo Webscope Data Sharing Agreement. Academic, non-commercial research requires a signed agreement. Yahoo prohibits redistribution.
     - Per-file SHA-256 (67 files). Every download must match its checksum.
     - Label 1 marks an anomaly point and label 0 marks a normal point. The value column is the only feature. Rows keep file order.

Yahoo S5 sources
----------------

``Data.datasets("yahoo")`` lists the 67 A1Benchmark series. The loader excludes the synthetic A2, A3, and A4 subsets.

The `Yahoo S5 catalog <https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70>`_ describes the original dataset. Downloads use a `public mirror at a fixed commit <https://github.com/ayyzenn/Real-Time-Anomaly-Detection-in-Kubernetes-Cluster-using-Three-Way-Decisions/tree/79453c498fef5bc23310682355e1dea1fb191b97/yahoo_data>`_. The mirror's `WebscopeReadMe.txt <https://github.com/ayyzenn/Real-Time-Anomaly-Detection-in-Kubernetes-Cluster-using-Three-Way-Decisions/blob/79453c498fef5bc23310682355e1dea1fb191b97/yahoo_data/WebscopeReadMe.txt>`_ states Yahoo's usage restrictions. The mirror does not grant a separate data license. Users must obtain Yahoo's signed Data Sharing Agreement before use.

Checksum verification
---------------------

PySAD computes the SHA-256 checksum of every downloaded file and compares it with the expected value before it caches the file. If the checksum does not match, the download fails with an error. For loaders with many files, each file has its own checksum. The full list of per-file checksums is in ``pysad/utils/_datasets.py``. Loaders that download a single archive or file show the checksum in the table above.

Labels
------

Label 1 always marks the anomaly class and label 0 marks the normal class, except where the table notes otherwise. The label format depends on the loader: anomaly windows for NAB, anomaly ranges for SMAP and MSL, a Class column for credit-card fraud, and a last binary column for TSB-drift.
