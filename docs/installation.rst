Installation
============


The PySAD framework can be installed via:


.. code-block:: bash

    pip install -U pysad


Alternatively, you can install the library directly using the source code in Github repository by:


.. code-block:: bash

    git clone https://github.com/selimfirat/pysad.git
    cd pysad
    pip install .


**Required Dependencies:**


* Python 3.9+
* numpy==2.0.2
* scikit-learn==1.5.2
* scipy==1.13.1
* pyod==1.1.0
* combo==0.1.3

**Optional Dependencies:**


* rrcf==0.4.3 (Only required for :class:`pysad.models.robust_random_cut_forest.RobustRandomCutForest`)
* PyNomaly==0.3.3 (Only required for :class:`pysad.models.loop.StreamLocalOutlierProbability`)
* mmh3==2.5.1 (Only required for :class:`pysad.models.xstream.xStream`)
* pandas==2.2.3 (Only required for :class:`pysad.utils.pandas_streamer.PandasStreamer`)
* jax==0.4.17 (Only required for :class:`pysad.models.inqmad.Inqmad`)
* jaxlib==0.4.17 (Only required for :class:`pysad.models.inqmad.Inqmad`)
