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


* numpy>=1.18.5
* scipy>=1.4.1
* scikit-learn>=0.23.2
* pyod>=0.7.7.1

**Optional Dependencies:**


* rrcf==0.4.3 (Only required for :class:`pysad.models.robust_random_cut_forest.RobustRandomCutForest`)
* PyNomaly==0.3.3 (Only required for :class:`pysad.models.loop.StreamLocalOutlierProbability`)
* mmh3==2.5.1 (Only required for :class:`pysad.models.xstream.xStream`)
* pandas==1.1.0 (Only required for :class:`pysad.utils.pandas_streamer.PandasStreamer`)
