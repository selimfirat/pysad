API reference
=============

Each page lists the public classes and functions of one module. Each name has a one-line summary. Import each name from the module that its page documents, for example ``from pysad.models import LODA``.

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Page
     - What it holds
     - Module
   * - :doc:`api/models`
     - Streaming anomaly detectors
     - ``pysad.models``
   * - :doc:`api/integrations`
     - Bidirectional wrappers and adapters for PyOD, scikit-learn, and River
     - ``pysad.integrations``
   * - :doc:`api/preprocessing`
     - Scalers and seasonal decomposition
     - ``pysad.transform.preprocessing``
   * - :doc:`api/projection`
     - Random projections
     - ``pysad.transform.projection``
   * - :doc:`api/stream_generators`
     - Adversarial stream generators
     - ``pysad.transform.stream_generators``
   * - :doc:`api/postprocessing`
     - Score smoothing and normalization
     - ``pysad.transform.postprocessing``
   * - :doc:`api/ensemble`
     - Score ensemblers
     - ``pysad.transform.ensemble``
   * - :doc:`api/calibration`
     - Score calibrators
     - ``pysad.transform.probability_calibration``
   * - :doc:`api/evaluation`
     - Streaming metrics, including :class:`~pysad.evaluation.RollingMetric`
     - ``pysad.evaluation``
   * - :doc:`api/statistics`
     - Running statistics
     - ``pysad.statistics``
   * - :doc:`api/utils`
     - Streamers, datasets and helpers
     - ``pysad.utils``
   * - :doc:`api/core`
     - Abstract base classes
     - ``pysad.core``

.. toctree::
   :hidden:

   api/models
   api/integrations
   api/preprocessing
   api/projection
   api/stream_generators
   api/postprocessing
   api/ensemble
   api/calibration
   api/evaluation
   api/statistics
   api/utils
   api/core
