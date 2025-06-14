.. image:: https://github.com/selimfirat/pysad/raw/master/docs/logo.png
    :align: center

.. include:: docs/badges.rst

.. include:: docs/features.rst

.. include:: docs/installation.rst

Examples
========

Quick Start
^^^^^^^^^^^^^^^^^^

Here's a simple example showing how to use PySAD for anomaly detection on streaming data:

.. code-block:: python

    # Import modules.
    from pysad.evaluation import AUROCMetric
    from pysad.models import LODA
    from pysad.utils import Data


    model = LODA()  # Init model
    metric = AUROCMetric()  # Init area under receiver-operating- characteristics curve metric
    streaming_data = Data().get_iterator("arrhythmia.mat")  # Get data streamer.

    for x, y_true in streaming_data:  # Stream data.
        anomaly_score = model.fit_score_partial(x)  # Fit the instance to model and score the instance.

        metric.update(y_true, anomaly_score)  # Update the AUROC metric.

    # Output the resulting AUROCMetric.
    print(f"Area under ROC metric is {metric.get()}.")

.. include:: docs/quick_links.rst

Contributors
============

.. raw:: html

   <div align="center">
     <a href="https://github.com/selimfirat/pysad/graphs/contributors">
       <img src="https://contrib.rocks/image?repo=selimfirat/pysad" alt="Contributors" />
     </a>
   </div>

We thank all our contributors for their valuable input and efforts to make PySAD better!

.. include:: docs/versioning.rst

.. include:: docs/license.rst

.. include:: docs/citing.rst
