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

* Python: 3.10+
* numpy: 2.1.3
* scikit-learn: 1.5.2
* scipy: 1.15.3
* pyod: 2.1.0
* combo: 0.1.3

**Optional Dependencies:**

* rrcf: 0.4.4 (for ``pysad.models.robust_random_cut_forest.RobustRandomCutForest``)
* PyNomaly: 0.3.5 (for ``pysad.models.loop.StreamLocalOutlierProbability``)
* mmh3: 2.5.1 (for ``pysad.models.xstream.xStream``)
* pandas: 2.2.3 (for ``pysad.utils.pandas_streamer.PandasStreamer``)
* jax: >=0.6.1 (for ``pysad.models.inqmad.Inqmad``; required for NumPy 2.0+ compatibility of this module)
* jaxlib: >=0.6.1 (for ``pysad.models.inqmad.Inqmad``; required for NumPy 2.0+ compatibility of this module)
